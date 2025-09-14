/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.viderlab.digitaltravelguide.kotlin.helloar

import android.media.Image
import android.opengl.GLES30
import android.opengl.Matrix
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Anchor
import com.google.ar.core.Camera
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.Coordinates2d
import com.google.ar.core.HitResult
import com.google.ar.core.InstantPlacementPoint
import com.google.ar.core.LightEstimate
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.Trackable
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.viderlab.digitaltravelguide.java.common.helpers.DisplayRotationHelper
import com.viderlab.digitaltravelguide.java.common.helpers.TrackingStateHelper
import com.viderlab.digitaltravelguide.java.common.samplerender.Framebuffer
import com.viderlab.digitaltravelguide.java.common.samplerender.GLError
import com.viderlab.digitaltravelguide.java.common.samplerender.Mesh
import com.viderlab.digitaltravelguide.java.common.samplerender.SampleRender
import com.viderlab.digitaltravelguide.java.common.samplerender.Shader
import com.viderlab.digitaltravelguide.java.common.samplerender.Texture
import com.viderlab.digitaltravelguide.java.common.samplerender.VertexBuffer
import com.viderlab.digitaltravelguide.java.common.samplerender.arcore.BackgroundRenderer
import com.viderlab.digitaltravelguide.java.common.samplerender.arcore.PlaneRenderer
import com.viderlab.digitaltravelguide.java.common.samplerender.arcore.SpecularCubemapFilter
import com.viderlab.digitaltravelguide.kotlin.inference.InferenceModule
import com.viderlab.digitaltravelguide.kotlin.inference.PipelineConfig
import com.viderlab.digitaltravelguide.kotlin.common.helpers.YuvToRgbConverter
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.DeadlineExceededException
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.jvm.Volatile
import kotlin.math.max
import kotlin.math.sqrt

data class BoundingBox(val x1: Float, val y1: Float, val x2: Float, val y2: Float)

/** Simple wrapper pairing a [BoundingBox] with its detection label and score. */
data class LabeledBoundingBox(val bbox: BoundingBox, val label: String, val score: Float)

/** Renders the HelloAR application using our example Renderer. */
class HelloArRenderer(val activity: HelloArActivity, initialPipeline: PipelineConfig) :
  SampleRender.Renderer, DefaultLifecycleObserver {
  companion object {
    val TAG = "HelloArRenderer"

    // See the definition of updateSphericalHarmonicsCoefficients for an explanation of these
    // constants.
    private val sphericalHarmonicFactors =
      floatArrayOf(
        0.282095f,
        -0.325735f,
        0.325735f,
        -0.325735f,
        0.273137f,
        -0.273137f,
        0.078848f,
        -0.273137f,
        0.136569f
      )

    private val Z_NEAR = 0.1f
    private val Z_FAR = 100f

    // Assumed distance from the device camera to the surface where objects will be placed.
    // This value affects the apparent scale of objects while the tracking method of the
    // Instant Placement point is SCREENSPACE_WITH_APPROXIMATE_DISTANCE.
    // Values in the [0.2, 2.0] meter range are a good choice for most AR experiences.
    // Use lower values when objects are placed on surfaces close to the camera and larger values
    // when objects are placed on the ground or floor in front of the camera.
    val APPROXIMATE_DISTANCE_METERS = 2.0f

    val CUBEMAP_RESOLUTION = 16
    val CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES = 32

    private const val MIN_SCALE = 0.1f
    private const val BASE_SCALE = 0.2f
  }

  lateinit var render: SampleRender
  lateinit var planeRenderer: PlaneRenderer
  lateinit var backgroundRenderer: BackgroundRenderer
  lateinit var virtualSceneFramebuffer: Framebuffer
  var hasSetTextureNames = false

  // Point Cloud
  lateinit var pointCloudVertexBuffer: VertexBuffer
  lateinit var pointCloudMesh: Mesh
  lateinit var pointCloudShader: Shader

  // Keep track of the last point cloud rendered to avoid updating the VBO if point cloud
  // was not changed.  Do this using the timestamp since we can't compare PointCloud objects.
  var lastPointCloudTimestamp: Long = 0

  // Virtual object (ARCore pawn)
  lateinit var virtualObjectMesh: Mesh
  lateinit var virtualObjectShader: Shader
  lateinit var virtualObjectAlbedoTexture: Texture
  lateinit var virtualObjectAlbedoInstantPlacementTexture: Texture

  private val anchorsByLabel = LinkedHashMap<String, WrappedAnchor>()

  private val AUTO_PLACE_INTERVAL_MS = 3_000L
  private var lastAutoPlaceTime = 0L

  // Environmental HDR
  lateinit var dfgTexture: Texture
  lateinit var cubemapFilter: SpecularCubemapFilter

  // Temporary matrix allocated here to reduce number of allocations for each frame.
  val modelMatrix = FloatArray(16)
  val viewMatrix = FloatArray(16)
  val projectionMatrix = FloatArray(16)
  val modelViewMatrix = FloatArray(16) // view x model

  val modelViewProjectionMatrix = FloatArray(16) // projection x view x model

  val sphericalHarmonicsCoefficients = FloatArray(9 * 3)
  val viewInverseMatrix = FloatArray(16)
  val worldLightDirection = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
  val viewLightDirection = FloatArray(4) // view x world light direction

  val session
    get() = activity.arCoreSessionHelper.session

  val displayRotationHelper = DisplayRotationHelper(activity)
  val trackingStateHelper = TrackingStateHelper(activity)
  private val inferenceModule = InferenceModule(activity, initialPipeline)
  private val yuvToRgbConverter = YuvToRgbConverter(activity)

  private val detectionExecutor = Executors.newSingleThreadExecutor()
  private val isImageProcessing = AtomicBoolean(false)
  private val detectionQueue = LinkedBlockingQueue<List<LabeledBoundingBox>>(1)
  @Volatile private var isActive = true

  fun setInferenceEnabled(enabled: Boolean) {
    isActive = enabled
  }

  fun isInferenceEnabled(): Boolean {
    return isActive
  }

  fun setPipeline(config: PipelineConfig) {
    // Temporarily disable detection to avoid running inference while
    // reconfiguring the pipeline. This prevents concurrent access to the
    // inference session while it is being closed and recreated.
    isActive = false

    // Wait for any ongoing image processing to finish before proceeding.
    // This ensures that `runInference` is not using the session when we
    // attempt to close it inside `inferenceModule.load`.
    while (isImageProcessing.get()) {
      try {
        Thread.sleep(5)
      } catch (e: InterruptedException) {
        Thread.currentThread().interrupt()
      }
    }

    // Execute the load on the same executor used for detections so that it
    // cannot overlap with `runInference`.
    detectionExecutor.submit {
      inferenceModule.load(config)
    }.get()

    // Reactivate detection now that the new session has been created.
    isActive = true
  }

  override fun onResume(owner: LifecycleOwner) {
    displayRotationHelper.onResume()
    hasSetTextureNames = false
    isActive = true
  }

  override fun onPause(owner: LifecycleOwner) {
    displayRotationHelper.onPause()
    isActive = false
  }

  override fun onSurfaceCreated(render: SampleRender) {
    // Prepare the rendering objects.
    // This involves reading shaders and 3D model files, so may throw an IOException.
    try {
      planeRenderer = PlaneRenderer(render)
      backgroundRenderer = BackgroundRenderer(render)
      virtualSceneFramebuffer = Framebuffer(render, /*width=*/ 1, /*height=*/ 1)

      cubemapFilter =
        SpecularCubemapFilter(render, CUBEMAP_RESOLUTION, CUBEMAP_NUMBER_OF_IMPORTANCE_SAMPLES)
      // Load environmental lighting values lookup table
      dfgTexture =
        Texture(
          render,
          Texture.Target.TEXTURE_2D,
          Texture.WrapMode.CLAMP_TO_EDGE,
          /*useMipmaps=*/ false
        )
      // The dfg.raw file is a raw half-float texture with two channels.
      val dfgResolution = 64
      val dfgChannels = 2
      val halfFloatSize = 2

      val buffer: ByteBuffer =
        ByteBuffer.allocateDirect(dfgResolution * dfgResolution * dfgChannels * halfFloatSize)
      activity.assets.open("models/dfg.raw").use { it.read(buffer.array()) }

      // SampleRender abstraction leaks here.
      GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, dfgTexture.textureId)
      GLError.maybeThrowGLException("Failed to bind DFG texture", "glBindTexture")
      GLES30.glTexImage2D(
        GLES30.GL_TEXTURE_2D,
        /*level=*/ 0,
        GLES30.GL_RG16F,
        /*width=*/ dfgResolution,
        /*height=*/ dfgResolution,
        /*border=*/ 0,
        GLES30.GL_RG,
        GLES30.GL_HALF_FLOAT,
        buffer
      )
      GLError.maybeThrowGLException("Failed to populate DFG texture", "glTexImage2D")

      // Point cloud
      pointCloudShader =
        Shader.createFromAssets(
            render,
            "shaders/point_cloud.vert",
            "shaders/point_cloud.frag",
            /*defines=*/ null
          )
          .setVec4("u_Color", floatArrayOf(31.0f / 255.0f, 188.0f / 255.0f, 210.0f / 255.0f, 1.0f))
          .setFloat("u_PointSize", 5.0f)

      // four entries per vertex: X, Y, Z, confidence
      pointCloudVertexBuffer =
        VertexBuffer(render, /*numberOfEntriesPerVertex=*/ 4, /*entries=*/ null)
      val pointCloudVertexBuffers = arrayOf(pointCloudVertexBuffer)
      pointCloudMesh =
        Mesh(render, Mesh.PrimitiveMode.POINTS, /*indexBuffer=*/ null, pointCloudVertexBuffers)

      // Virtual object to render (ARCore pawn)
      virtualObjectAlbedoTexture =
        Texture.createFromAsset(
          render,
          "models/pawn_albedo.png",
          Texture.WrapMode.CLAMP_TO_EDGE,
          Texture.ColorFormat.SRGB
        )

      virtualObjectAlbedoInstantPlacementTexture =
        Texture.createFromAsset(
          render,
          "models/pawn_albedo_instant_placement.png",
          Texture.WrapMode.CLAMP_TO_EDGE,
          Texture.ColorFormat.SRGB
        )

      val virtualObjectPbrTexture =
        Texture.createFromAsset(
          render,
          "models/pawn_roughness_metallic_ao.png",
          Texture.WrapMode.CLAMP_TO_EDGE,
          Texture.ColorFormat.LINEAR
        )
      virtualObjectMesh = Mesh.createFromAsset(render, "models/pawn.obj")
      virtualObjectShader =
        Shader.createFromAssets(
            render,
            "shaders/environmental_hdr.vert",
            "shaders/environmental_hdr.frag",
            mapOf("NUMBER_OF_MIPMAP_LEVELS" to cubemapFilter.numberOfMipmapLevels.toString())
          )
          .setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture)
          .setTexture("u_RoughnessMetallicAmbientOcclusionTexture", virtualObjectPbrTexture)
          .setTexture("u_Cubemap", cubemapFilter.filteredCubemapTexture)
          .setTexture("u_DfgTexture", dfgTexture)
    } catch (e: IOException) {
      Log.e(TAG, "Failed to read a required asset file", e)
      showError("Failed to read a required asset file: $e")
    }
  }

  override fun onSurfaceChanged(render: SampleRender, width: Int, height: Int) {
    displayRotationHelper.onSurfaceChanged(width, height)
    virtualSceneFramebuffer.resize(width, height)
  }

  override fun onDrawFrame(render: SampleRender) {
    val session = session ?: return

    // Texture names should only be set once on a GL thread unless they change. This is done during
    // onDrawFrame rather than onSurfaceCreated since the session is not guaranteed to have been
    // initialized during the execution of onSurfaceCreated.
    if (!hasSetTextureNames) {
      session.setCameraTextureNames(intArrayOf(backgroundRenderer.cameraColorTexture.textureId))
      hasSetTextureNames = true
    }

    // -- Update per-frame state

    // Notify ARCore session that the view size changed so that the perspective matrix and
    // the video background can be properly adjusted.
    displayRotationHelper.updateSessionIfNeeded(session)

    // Obtain the current frame from ARSession. When the configuration is set to
    // UpdateMode.BLOCKING (it is by default), this will throttle the rendering to the
    // camera framerate.
    val frame =
      try {
        session.update()
      } catch (e: CameraNotAvailableException) {
        Log.e(TAG, "Camera not available during onDrawFrame", e)
        showError("Camera not available. Try restarting the app.")
        return
      }

    val camera = frame.camera

    // Update BackgroundRenderer state to match the depth settings.
    try {
      backgroundRenderer.setUseDepthVisualization(
        render,
        activity.depthSettings.depthColorVisualizationEnabled()
      )
      backgroundRenderer.setUseOcclusion(render, activity.depthSettings.useDepthForOcclusion())
    } catch (e: IOException) {
      Log.e(TAG, "Failed to read a required asset file", e)
      showError("Failed to read a required asset file: $e")
      return
    }

    // BackgroundRenderer.updateDisplayGeometry must be called every frame to update the coordinates
    // used to draw the background camera image.
    backgroundRenderer.updateDisplayGeometry(frame)
    val shouldGetDepthImage =
      activity.depthSettings.useDepthForOcclusion() ||
        activity.depthSettings.depthColorVisualizationEnabled()
    if (camera.trackingState == TrackingState.TRACKING && shouldGetDepthImage) {
      try {
        val depthImage = frame.acquireDepthImage16Bits()
        backgroundRenderer.updateCameraDepthTexture(depthImage)
        depthImage.close()
      } catch (e: NotYetAvailableException) {
        // This normally means that depth data is not available yet. This is normal so we will not
        // spam the logcat with this.
      }
    }

    fetchDetections(frame)
    maybePlaceAutoMarker(frame, camera)

    // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
    trackingStateHelper.updateKeepScreenOnFlag(camera.trackingState)

    // Show a message based on whether tracking has failed, if planes are detected, and whether any
    // objects have been placed.
    val message: String? =
      when {
        camera.trackingState == TrackingState.PAUSED &&
          camera.trackingFailureReason == TrackingFailureReason.NONE ->
          activity.getString(R.string.searching_planes)
        camera.trackingState == TrackingState.PAUSED ->
          TrackingStateHelper.getTrackingFailureReasonString(camera)
        session.hasTrackingPlane() -> null
        else -> activity.getString(R.string.searching_planes)
      }
    if (message == null) {
      activity.view.snackbarHelper.hide(activity)
    } else {
      activity.view.snackbarHelper.showMessage(activity, message)
    }

    // -- Draw background
    if (frame.timestamp != 0L) {
      // Suppress rendering if the camera did not produce the first frame yet. This is to avoid
      // drawing possible leftover data from previous sessions if the texture is reused.
      backgroundRenderer.drawBackground(render)
    }

    // If not tracking, don't draw 3D objects.
    if (camera.trackingState == TrackingState.PAUSED) {
      return
    }

    // -- Draw non-occluded virtual objects (planes, point cloud)

    // Get projection matrix.
    camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR)

    // Get camera matrix and draw.
    camera.getViewMatrix(viewMatrix, 0)
    if (activity.planePointSettings.showPlanesAndPoints) {
      try {
        frame.acquirePointCloud().use { pointCloud ->
          if (pointCloud.timestamp > lastPointCloudTimestamp) {
            pointCloudVertexBuffer.set(pointCloud.points)
            lastPointCloudTimestamp = pointCloud.timestamp
          }
          Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
          pointCloudShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
          render.draw(pointCloudMesh, pointCloudShader)
        }
      } catch (e: DeadlineExceededException) {
        Log.w(TAG, "Point cloud acquisition deadline exceeded", e)
        return
      }

      // Visualize planes.
      planeRenderer.drawPlanes(
        render,
        session.getAllTrackables<Plane>(Plane::class.java),
        camera.displayOrientedPose,
        projectionMatrix
      )
    }

    // -- Draw occluded virtual objects

    // Update lighting parameters in the shader
    updateLightEstimation(frame.lightEstimate, viewMatrix)

    // Visualize anchors created automatically from detected markers.
    render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f)
    val infoList = mutableListOf<MarkerInfo>()
    for (wrapped in anchorsByLabel.values.filter { it.anchor.trackingState == TrackingState.TRACKING }) {
      val anchor = wrapped.anchor
      val trackable = wrapped.trackable
      // Get the current pose of an Anchor in world space. The Anchor pose is updated
      // during calls to session.update() as ARCore refines its estimate of the world.
      anchor.pose.toMatrix(modelMatrix, 0)

      val cameraPose = camera.pose
      val anchorPose = anchor.pose
      val dx = cameraPose.tx() - anchorPose.tx()
      val dy = cameraPose.ty() - anchorPose.ty()
      val dz = cameraPose.tz() - anchorPose.tz()
      val distance = sqrt(dx * dx + dy * dy + dz * dz)

      val scale = max(MIN_SCALE, BASE_SCALE * distance)
      Matrix.scaleM(modelMatrix, 0, scale, scale, scale)

      // Calculate model/view/projection matrices
      Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)
      Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)

      // Compute screen coordinates
      val clipCoords = FloatArray(4)
      Matrix.multiplyMV(
        clipCoords,
        0,
        modelViewProjectionMatrix,
        0,
        floatArrayOf(0f, 0f, 0f, 1f),
        0
      )
      val ndcX = clipCoords[0] / clipCoords[3]
      val ndcY = clipCoords[1] / clipCoords[3]
      val viewWidth = activity.view.surfaceView.width.toFloat()
      val viewHeight = activity.view.surfaceView.height.toFloat()
      val screenX = ((ndcX + 1f) / 2f) * viewWidth
      val screenY = ((1f - ndcY) / 2f) * viewHeight

      val scaledBbox = wrapped.bbox?.let { bbox ->
        val scaleFactor = distance / wrapped.initialDistance
        val width = (bbox.x2 - bbox.x1) * scaleFactor
        val height = (bbox.y2 - bbox.y1) * scaleFactor
        BoundingBox(
          screenX - width / 2f,
          screenY - height / 2f,
          screenX + width / 2f,
          screenY + height / 2f
        )
      }
      infoList += MarkerInfo(screenX, screenY, wrapped.label, scaledBbox)

      // Update shader properties and draw
      virtualObjectShader.setMat4("u_ModelView", modelViewMatrix)
      virtualObjectShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
      val texture =
        if ((trackable as? InstantPlacementPoint)?.trackingMethod ==
            InstantPlacementPoint.TrackingMethod.SCREENSPACE_WITH_APPROXIMATE_DISTANCE
        ) {
          virtualObjectAlbedoInstantPlacementTexture
        } else {
          virtualObjectAlbedoTexture
        }
      virtualObjectShader.setTexture("u_AlbedoTexture", texture)
      render.draw(virtualObjectMesh, virtualObjectShader, virtualSceneFramebuffer)
    }
    activity.runOnUiThread { activity.view.updateAnchors(infoList) }

    // Compose the virtual scene with the background.
    backgroundRenderer.drawVirtualScene(render, virtualSceneFramebuffer, Z_NEAR, Z_FAR)
  }

  /** Checks if we detected at least one plane. */
  private fun Session.hasTrackingPlane() =
    getAllTrackables(Plane::class.java).any { it.trackingState == TrackingState.TRACKING }

  /** Update state based on the current frame's light estimation. */
  private fun updateLightEstimation(lightEstimate: LightEstimate, viewMatrix: FloatArray) {
    if (lightEstimate.state != LightEstimate.State.VALID) {
      virtualObjectShader.setBool("u_LightEstimateIsValid", false)
      return
    }
    virtualObjectShader.setBool("u_LightEstimateIsValid", true)
    Matrix.invertM(viewInverseMatrix, 0, viewMatrix, 0)
    virtualObjectShader.setMat4("u_ViewInverse", viewInverseMatrix)
    updateMainLight(
      lightEstimate.environmentalHdrMainLightDirection,
      lightEstimate.environmentalHdrMainLightIntensity,
      viewMatrix
    )
    updateSphericalHarmonicsCoefficients(lightEstimate.environmentalHdrAmbientSphericalHarmonics)
    cubemapFilter.update(lightEstimate.acquireEnvironmentalHdrCubeMap())
  }

  private fun updateMainLight(
    direction: FloatArray,
    intensity: FloatArray,
    viewMatrix: FloatArray
  ) {
    // We need the direction in a vec4 with 0.0 as the final component to transform it to view space
    worldLightDirection[0] = direction[0]
    worldLightDirection[1] = direction[1]
    worldLightDirection[2] = direction[2]
    Matrix.multiplyMV(viewLightDirection, 0, viewMatrix, 0, worldLightDirection, 0)
    virtualObjectShader.setVec4("u_ViewLightDirection", viewLightDirection)
    virtualObjectShader.setVec3("u_LightIntensity", intensity)
  }

  private fun updateSphericalHarmonicsCoefficients(coefficients: FloatArray) {
    // Pre-multiply the spherical harmonics coefficients before passing them to the shader. The
    // constants in sphericalHarmonicFactors were derived from three terms:
    //
    // 1. The normalized spherical harmonics basis functions (y_lm)
    //
    // 2. The lambertian diffuse BRDF factor (1/pi)
    //
    // 3. A <cos> convolution. This is done to so that the resulting function outputs the irradiance
    // of all incoming light over a hemisphere for a given surface normal, which is what the shader
    // (environmental_hdr.frag) expects.
    //
    // You can read more details about the math here:
    // https://google.github.io/filament/Filament.html#annex/sphericalharmonics
    require(coefficients.size == 9 * 3) {
      "The given coefficients array must be of length 27 (3 components per 9 coefficients"
    }

    // Apply each factor to every component of each coefficient
    for (i in 0 until 9 * 3) {
      sphericalHarmonicsCoefficients[i] = coefficients[i] * sphericalHarmonicFactors[i / 3]
    }
    virtualObjectShader.setVec3Array(
      "u_SphericalHarmonicsCoefficients",
      sphericalHarmonicsCoefficients
    )
  }

  fun fetchDetections(frame: Frame) {
    if (!isActive) return
    if (!isImageProcessing.compareAndSet(false, true)) return
    Log.d(TAG, "Entering fetchDetections")
    var image: Image? = null
    try {
      Log.d(TAG, "Calling acquireCameraImage")
      image = frame.acquireCameraImage()
      Log.d(TAG, "acquireCameraImage completed")

      Log.d(TAG, "Starting YUV to RGB conversion")
      val bitmap = yuvToRgbConverter.yuvToRgb(image)
      Log.d(TAG, "YUV to RGB conversion completed")
      image.close()
      image = null

      val viewWidth = activity.view.surfaceView.width.toFloat()
      val viewHeight = activity.view.surfaceView.height.toFloat()

      detectionExecutor.execute {
        try {
          if (!isActive) return@execute

          val sx = viewWidth / bitmap.width.toFloat()
          val sy = viewHeight / bitmap.height.toFloat()

          Log.d(TAG, "Running inference")
          val detections =
            inferenceModule.runInference(bitmap).let { results ->
              if (activity.boundingBoxSettings.showTestDetection) {
                results
              } else {
                results.filter { it.label != "Test" }
              }
            }
          Log.d(TAG, "Inference completed with ${detections.size} detections")

          if (!isActive) return@execute

          val labeled = detections.map { d ->
            Log.d(TAG, "Detected ${d.label} at ${d.box} score=${d.score}")
            val bbox = BoundingBox(
              d.box.left.toFloat() * sx,
              d.box.top.toFloat() * sy,
              d.box.right.toFloat() * sx,
              d.box.bottom.toFloat() * sy
            )
            LabeledBoundingBox(bbox, d.label, d.score)
          }
          detectionQueue.clear()
          if (labeled.isNotEmpty()) {
            detectionQueue.offer(labeled)
          }
          activity.runOnUiThread {
            if (activity.boundingBoxSettings.showDetectionList) {
              activity.view.showDetectionList(labeled)
            } else {
              activity.view.clearDetectionList()
            }
          }
          if (activity.boundingBoxSettings.showBoundingBoxes) {
            activity.runOnUiThread { activity.view.showDebugBoundingBoxes(labeled) }
          }
        } finally {
          isImageProcessing.set(false)
        }
      }
    } catch (e: NotYetAvailableException) {
      Log.w(TAG, "Camera image not ready", e)
      isImageProcessing.set(false)
    } finally {
      image?.close()
    }
  }

  /**
   * Generates a grid of candidate points inside a bounding box.
   *
   * The grid intentionally leaves a small margin on each side to avoid
   * sampling right on the edges which are more likely to fail a hit test.
   */
  private fun generateCandidatePoints(bbox: BoundingBox, gridSize: Int = 3): List<Pair<Float, Float>> {
    val points = mutableListOf<Pair<Float, Float>>()
    val stepX = (bbox.x2 - bbox.x1) / (gridSize + 1)
    val stepY = (bbox.y2 - bbox.y1) / (gridSize + 1)
    for (i in 1..gridSize) {
      val x = bbox.x1 + i * stepX
      for (j in 1..gridSize) {
        val y = bbox.y1 + j * stepY
        points.add(Pair(x, y))
      }
    }
    return points
  }

  /**
   * Returns the first valid hit result for the given screen point, or null if none was found.
   */
  private fun firstHitResult(frame: Frame, camera: Camera, x: Float, y: Float): HitResult? {
    val hitResultList = frame.hitTest(x, y)
    return selectValidHit(hitResultList, camera)
  }

  private fun firstInstantHitResult(frame: Frame, camera: Camera, x: Float, y: Float): HitResult? {
    val hitResultList = frame.hitTestInstantPlacement(x, y, APPROXIMATE_DISTANCE_METERS)
    return selectValidHit(hitResultList, camera)
  }

  private fun approximateAnchorFromScreenPos(
    camera: Camera,
    x: Float,
    y: Float,
  ): Anchor {
    // This method performs only math operations. It never acquires images, therefore
    // it never blocks the rendering thread nor requires resource cleanup.

    // Obtain the projection matrix and invert it to transform screen points into view space.
    val projectionMatrix = FloatArray(16)
    camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR)
    val inverseProj = FloatArray(16)
    Matrix.invertM(inverseProj, 0, projectionMatrix, 0)

    // Convert screen coordinates into Normalized Device Coordinates (NDC).
    val viewWidth = activity.view.surfaceView.width.toFloat()
    val viewHeight = activity.view.surfaceView.height.toFloat()
    val xNdc = (x / viewWidth) * 2f - 1f
    val yNdc = 1f - (y / viewHeight) * 2f
    val clipCoords = floatArrayOf(xNdc, yNdc, 1f, 1f)

    // Generate a ray in view space pointing towards the selected screen position.
    val out = FloatArray(4)
    Matrix.multiplyMV(out, 0, inverseProj, 0, clipCoords, 0)
    val rayDir = floatArrayOf(out[0] / out[3], out[1] / out[3], out[2] / out[3])

    // Normalize the ray and scale it by the approximate distance to obtain a point in
    // camera space.
    val length =
      Math.sqrt(
        (rayDir[0] * rayDir[0] + rayDir[1] * rayDir[1] + rayDir[2] * rayDir[2]).toDouble()
      ).toFloat()
    val normalizedRay = floatArrayOf(rayDir[0] / length, rayDir[1] / length, rayDir[2] / length)
    val pointCam = floatArrayOf(
      normalizedRay[0] * APPROXIMATE_DISTANCE_METERS,
      normalizedRay[1] * APPROXIMATE_DISTANCE_METERS,
      normalizedRay[2] * APPROXIMATE_DISTANCE_METERS,
    )

    // Compose the translation with the camera pose and create an anchor at that position.
    val pose = camera.pose.compose(Pose.makeTranslation(pointCam[0], pointCam[1], pointCam[2]))
    val uprightPose = Pose.makeTranslation(pose.tx(), pose.ty(), pose.tz())
    return session!!.createAnchor(uprightPose)
  }

  private fun readDepthMeters(depthImage: Image, x: Int, y: Int): Float? {
    // Safely read a DEPTH16 pixel in meters. Stride is respected and the image is always
    // closed before returning to avoid stalling the renderer.
    return try {
      if (x < 0 || x >= depthImage.width || y < 0 || y >= depthImage.height) return null
      val plane = depthImage.planes[0]
      val buffer = plane.buffer
      val index = y * plane.rowStride + x * plane.pixelStride
      if (index + 1 >= buffer.capacity()) return null

      val depthMillimeters = buffer.getShort(index).toInt() and 0xFFFF
      if (depthMillimeters == 0) return null
      depthMillimeters / 1000f
    } finally {
      depthImage.close()
    }
  }

  private fun selectValidHit(hitResultList: List<HitResult>, camera: Camera): HitResult? =
    hitResultList.firstOrNull { hit ->
      when (val trackable = hit.trackable!!) {
        is Plane ->
          trackable.isPoseInPolygon(hit.hitPose) &&
            PlaneRenderer.calculateDistanceToPlane(hit.hitPose, camera.pose) > 0
        is Point -> trackable.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL
        is InstantPlacementPoint -> true
        // DepthPoints are only returned if Config.DepthMode is set to AUTOMATIC.
        is DepthPoint -> true
        else -> false
      }
    }

  private fun depthAnchorForBoundingBox(
    frame: Frame,
    camera: Camera,
    bbox: BoundingBox,
  ): Anchor? {
    // Acquires a depth image for a single pixel and releases it immediately, ensuring that
    // rendering is never blocked waiting for image resources.
    return try {
      val centerX = (bbox.x1 + bbox.x2) / 2f
      val centerY = (bbox.y1 + bbox.y2) / 2f

      val viewCoords = floatArrayOf(centerX, centerY)
      val depthCoords = FloatArray(2)
      frame.transformCoordinates2d(
        Coordinates2d.VIEW,
        viewCoords,
        Coordinates2d.IMAGE_PIXELS,
        depthCoords,
      )

      val depthX = depthCoords[0].toInt()
      val depthY = depthCoords[1].toInt()

      val depthMeters = readDepthMeters(frame.acquireDepthImage16Bits(), depthX, depthY)
      if (depthMeters == null) return null

      val projectionMatrix = FloatArray(16)
      camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR)
      val inverseProj = FloatArray(16)
      Matrix.invertM(inverseProj, 0, projectionMatrix, 0)

      val viewWidth = activity.view.surfaceView.width.toFloat()
      val viewHeight = activity.view.surfaceView.height.toFloat()
      val xNdc = (centerX / viewWidth) * 2f - 1f
      val yNdc = 1f - (centerY / viewHeight) * 2f
      val clipCoords = floatArrayOf(xNdc, yNdc, 1f, 1f)
      val out = FloatArray(4)
      Matrix.multiplyMV(out, 0, inverseProj, 0, clipCoords, 0)
      val ray = floatArrayOf(out[0] / out[3], out[1] / out[3], out[2] / out[3])
      val scale = depthMeters / -ray[2]
      val pointCam = floatArrayOf(ray[0] * scale, ray[1] * scale, -depthMeters)

      val pose =
        camera.pose.compose(
          Pose.makeTranslation(pointCam[0], pointCam[1], pointCam[2])
        )
      val uprightPose = Pose.makeTranslation(pose.tx(), pose.ty(), pose.tz())
      session!!.createAnchor(uprightPose)
    } catch (e: NotYetAvailableException) {
      null
    }
  }

  /**
   * Automatically places markers for each detected bounding box. For every box a
   * grid of candidate points is generated and hit tests are performed on each
   * point until a valid pose is found. If all candidates fail and instant
   * placement is enabled, the center of the bounding box is tested using
   * [Frame.hitTestInstantPlacement].
   */
  private fun maybePlaceAutoMarker(frame: Frame, camera: Camera) {
    val detections = detectionQueue.poll()
    if (activity.boundingBoxSettings.showBoundingBoxes && detections != null) {
      activity.runOnUiThread { activity.view.showDebugBoundingBoxes(detections) }
    }

    if (detections == null) return

    val time = System.currentTimeMillis()
    if (time - lastAutoPlaceTime < AUTO_PLACE_INTERVAL_MS) return

    for (detection: LabeledBoundingBox in detections) {
      val bbox = detection.bbox
      val centerX = (bbox.x1 + bbox.x2) / 2f
      val centerY = (bbox.y1 + bbox.y2) / 2f

      val depthAnchor = depthAnchorForBoundingBox(frame, camera, bbox)
      if (depthAnchor != null) {
        placeMarkerAt(camera, depthAnchor, detection.label, bbox)
        continue
      }

      var hitResult: HitResult? = null
      for ((x, y) in generateCandidatePoints(bbox)) {
        hitResult = firstHitResult(frame, camera, x, y)
        if (hitResult != null) break
      }

      if (hitResult == null && activity.instantPlacementSettings.isInstantPlacementEnabled) {
        hitResult = firstInstantHitResult(frame, camera, centerX, centerY)
      }

      if (hitResult != null) {
        placeMarkerAt(camera, hitResult, detection.label, bbox)
      } else {
        val anchor = approximateAnchorFromScreenPos(camera, centerX, centerY)
        placeMarkerAt(camera, anchor, detection.label, bbox)
      }
    }
    lastAutoPlaceTime = time
  }

  private fun placeMarkerAt(
    camera: Camera,
    hitResult: HitResult,
    label: String,
    bbox: BoundingBox? = null,
  ) {
    val pose = hitResult.hitPose
    val uprightPose = Pose.makeTranslation(pose.tx(), pose.ty(), pose.tz())
    val anchor = session!!.createAnchor(uprightPose)
    val cameraPose = camera.pose
    val anchorPose = anchor.pose
    val dx = cameraPose.tx() - anchorPose.tx()
    val dy = cameraPose.ty() - anchorPose.ty()
    val dz = cameraPose.tz() - anchorPose.tz()
    val distance = sqrt(dx * dx + dy * dy + dz * dz)
    placeMarkerAt(camera, anchor, hitResult.trackable, label, bbox, distance)
  }

  private fun placeMarkerAt(
    camera: Camera,
    anchor: Anchor,
    label: String,
    bbox: BoundingBox? = null,
  ) {
    val cameraPose = camera.pose
    val anchorPose = anchor.pose
    val dx = cameraPose.tx() - anchorPose.tx()
    val dy = cameraPose.ty() - anchorPose.ty()
    val dz = cameraPose.tz() - anchorPose.tz()
    val distance = sqrt(dx * dx + dy * dy + dz * dz)
    placeMarkerAt(camera, anchor, null, label, bbox, distance)
  }

  private fun placeMarkerAt(
    camera: Camera,
    anchor: Anchor,
    trackable: Trackable?,
    label: String,
    bbox: BoundingBox?,
    initialDistance: Float,
  ) {
    if (camera.trackingState != TrackingState.TRACKING) return

    anchorsByLabel[label]?.let { existing ->
      existing.anchor.detach()
      anchorsByLabel.remove(label)
    } ?: run {
      if (anchorsByLabel.size >= 20) {
        anchorsByLabel.entries.first().let {
          it.value.anchor.detach()
          anchorsByLabel.remove(it.key)
        }
      }
    }

    anchorsByLabel[label] = WrappedAnchor(anchor, trackable, label, bbox, initialDistance)

    // For devices that support the Depth API, shows a dialog to suggest enabling
    // depth-based occlusion. This dialog needs to be spawned on the UI thread.
    activity.runOnUiThread { activity.view.showOcclusionDialogIfNeeded() }
  }

  fun clearMarkers() {
    anchorsByLabel.values.forEach { it.anchor.detach() }
    anchorsByLabel.clear()
    activity.runOnUiThread { activity.view.updateAnchors(emptyList()) }
  }

  private fun showError(errorMessage: String) =
    activity.view.snackbarHelper.showError(activity, errorMessage)
}

/**
 * Associates an Anchor with the trackable it was attached to, if any. This is used to be able to
 * check whether or not an Anchor originally was attached to an {@link InstantPlacementPoint}.
*/
private data class WrappedAnchor(
  val anchor: Anchor,
  val trackable: Trackable?,
  val label: String,
  val bbox: BoundingBox?,
  val initialDistance: Float,
)
