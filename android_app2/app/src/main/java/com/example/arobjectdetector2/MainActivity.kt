package com.example.arobjectdetector2

import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import android.media.Image
import android.widget.FrameLayout
import android.widget.ImageView
import android.view.WindowManager
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.arcore.createAnchorOrNull
import io.github.sceneview.loaders.ModelLoader
import io.github.sceneview.node.ModelNode
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import com.google.ar.core.Config
import android.opengl.Matrix
import java.nio.ByteOrder
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.view.View
import java.nio.ByteBuffer
import kotlin.io.use

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val USE_STATIC_FRAME = false
        /** Pixel distance threshold to recreate the AR anchor. */
        private const val ANCHOR_MOVE_THRESHOLD_PX = 50f

        // Cache the descriptor database in memory so it is loaded only once
        private var descriptorBytes: ByteArray? = null

    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: BoxOverlay
    private lateinit var sceneView: ARSceneView
    /**
     * Loader used to read 3-D assets from the app assets directory.
     * A single instance is kept to create model nodes when anchors are placed.
     */
    private lateinit var modelLoader: ModelLoader
    private lateinit var cameraExecutor: ExecutorService
    private val analyzer = YoloAnalyzer()
    private var ortSession: OrtSession? = null
    private var detector: YoloDetector? = null  // ← Nuevo
    private var staticBitmap: Bitmap? = null
    private var lastDetections: List<Detection> = emptyList()
    /** Currently placed anchor in the scene. */
    private var currentAnchorNode: io.github.sceneview.ar.node.AnchorNode? = null
    /** Detection used to place the currentAnchorNode. */
    private var currentAnchorDetection: Detection? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlay     = findViewById(R.id.overlay)
        sceneView   = findViewById(R.id.sceneView)
        sceneView.lifecycle = lifecycle
        sceneView.session?.let { session ->
            val config = Config(session).apply {
                planeFindingMode = Config.PlaneFindingMode.DISABLED
                depthMode = Config.DepthMode.AUTOMATIC
            }
            session.configure(config)
        }
        // Initialize the loader that will create 3-D model instances
        modelLoader = ModelLoader(sceneView.engine, this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        sceneView.onSessionUpdated = { _, frame ->
            cameraExecutor.execute { analyzer.analyze(frame) }
        }

        // DEBUG: carga la imagen fija solo una vez
        if (USE_STATIC_FRAME) {
            staticBitmap = try {
                assets.open("debug2.jpg").use { BitmapFactory.decodeStream(it) }
            } catch (e: Exception) {
                Log.e(TAG, "Error cargando imagen estática", e)
                null
            }

            staticBitmap?.let { bmp ->
                Log.d(TAG, "🔒 Static debug frame size: ${bmp.width}×${bmp.height}")

                // 2) Oculta la vista de cámara
                previewView.visibility = View.GONE

                // 3) Añade un ImageView al rootLayout con la imagen fija
                val iv = ImageView(this).apply {
                    layoutParams = FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                    )
                    scaleType = ImageView.ScaleType.CENTER_CROP
                    setImageBitmap(bmp)
                }
                findViewById<FrameLayout>(R.id.rootLayout).addView(iv, 0)
            } ?: Log.e(TAG, "No se pudo cargar la imagen estática")
        } else {
            // Hide the CameraX preview since we rely on ARCore camera feed
            previewView.visibility = View.GONE
        }

        loadDnnModel()

        previewView.post {
            // espera a que previewView (y overlay) tengan width/height válidos
            detector?.let { det ->
                staticBitmap?.let { bmp ->
                    val vw = if (USE_STATIC_FRAME) overlay.width else sceneView.width
                    val vh = if (USE_STATIC_FRAME) overlay.height else sceneView.height
                    if (vw > 0 && vh > 0) {
                        val viewDets = det.detectOnView(
                            bmp,
                            vw,
                            vh
                        )
                        overlay.setDetections(viewDets)
                    }
                }
            }
        }

        val metrics = resources.displayMetrics
        Log.d(TAG, "📐 Pantalla completa: ${metrics.widthPixels}×${metrics.heightPixels}")
    }



    private fun loadDnnModel() {
        Thread {
            try {
                val onnxName = "pipeline.onnx"
                val tmpFile = File(cacheDir, onnxName)
                if (!tmpFile.exists()) {
                    assets.open(onnxName).use { input ->
                        tmpFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }

                if (descriptorBytes == null) {
                    val dbFileName = "places_db.bin"
                    descriptorBytes = assets.open(dbFileName).use { it.readBytes() }
                }

                val env = OrtEnvironment.getEnvironment()
                ortSession = env.createSession(tmpFile.absolutePath, OrtSession.SessionOptions())
                Log.d(TAG, "ORT session loaded correctly")

                // Inicializa el detector
                val dbStream = java.io.ByteArrayInputStream(descriptorBytes!!)
                detector = YoloDetector(
                    this,
                    session = ortSession!!,
                    descriptorsStream = dbStream
                )
            } catch (e: Exception) {
            Log.e(TAG, "Error cargando el modelo ONNX", e)
            runOnUiThread {
                Toast.makeText(
                    this,
                    "Error cargando el modelo: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }.start()
    }

    override fun onResume() {
        super.onResume()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onPause() {
        super.onPause()
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
    override fun onDestroy() {
        super.onDestroy()
        clearMarkers()
        cameraExecutor.shutdown()
        detector?.close()
        ortSession?.close()
    }

    private fun clearMarkers() {
        currentAnchorNode?.let { node ->
            sceneView.removeChildNode(node)
            node.destroy()
            currentAnchorNode = null
            currentAnchorDetection = null
        }
    }

    private fun placeMarker(det: Detection) {
        // Obtain the latest ARCore frame already handled by ARSceneView
        val frame = sceneView.session?.frame ?: return
        if (frame.camera.trackingState != TrackingState.TRACKING) {
            Log.w(TAG, "Cannot place marker: camera tracking state is ${frame.camera.trackingState}")
            return
        }
        val centerX = det.box.centerX()
        val centerY = det.box.centerY()

        // Try to read depth at the detection center
        val depthImage = runCatching { frame.acquireDepthImage() }.getOrNull()
        val anchor = depthImage?.use { img ->
            val px = (centerX / sceneView.width * img.width).toInt()
            val py = (centerY / sceneView.height * img.height).toInt()
            val plane = img.planes[0]
            val buffer = plane.buffer.order(ByteOrder.LITTLE_ENDIAN)
            val index = py * plane.rowStride + px * plane.pixelStride
            if (index >= 0 && index + 2 <= buffer.capacity()) {
                val depthMm = buffer.getShort(index).toInt() and 0xFFFF
                val depth = depthMm / 1000f
                if (depth > 0f) {
                    val proj = FloatArray(16).also { frame.camera.getProjectionMatrix(it, 0, 0.1f, 100f) }
                    val view = FloatArray(16).also { frame.camera.getViewMatrix(it, 0) }
                    val world = screenToWorld(centerX, centerY, depth, proj, view, sceneView.width, sceneView.height)
                    val pose = Pose.makeTranslation(world[0], world[1], world[2])
                    sceneView.session?.createAnchor(pose)
                } else null
            } else null
        } ?: run {
            // Fallback to hit-test based anchoring
            val hit = frame.hitTest(centerX, centerY).firstOrNull()
            if (hit != null) {
                hit.createAnchorOrNull()
            } else {
                val camPose = frame.camera.pose
                val pose = camPose.compose(Pose.makeTranslation(0f, 0f, -1f))
                sceneView.session?.createAnchor(pose)
            }
        } ?: return

        // Remove previous anchor if present
        currentAnchorNode?.let { node ->
            sceneView.removeChildNode(node)
            node.destroy()
            currentAnchorNode = null
        }

        val anchorNode = io.github.sceneview.ar.node.AnchorNode(sceneView.engine, anchor)

        // Load the marker model and attach it to the new anchor
        val modelInstance = modelLoader.createModelInstance("location.glb")
        val modelNode = ModelNode(modelInstance)
        anchorNode.addChildNode(modelNode)
        sceneView.addChildNode(anchorNode)
        currentAnchorNode = anchorNode
        currentAnchorDetection = det
    }

    private fun cameraImageToBitmap(image: Image): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            90,
            out
        )
        val bmp = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
        return bmp
    }

    private fun screenToWorld(
        x: Float,
        y: Float,
        depth: Float,
        projection: FloatArray,
        view: FloatArray,
        width: Int,
        height: Int
    ): FloatArray {
        val viewProj = FloatArray(16)
        val inverted = FloatArray(16)
        Matrix.multiplyMM(viewProj, 0, projection, 0, view, 0)
        Matrix.invertM(inverted, 0, viewProj, 0)

        val ndcX = (2f * x / width) - 1f
        val ndcY = 1f - 2f * y / height
        val near = 0.1f
        val far = 100f
        val ndcZ = (2f * depth - near - far) / (far - near)
        val clip = floatArrayOf(ndcX, ndcY, ndcZ, 1f)
        val world = FloatArray(4)
        Matrix.multiplyMV(world, 0, inverted, 0, clip, 0)
        val w = world[3]
        return floatArrayOf(world[0] / w, world[1] / w, world[2] / w)
    }


    private inner class YoloAnalyzer {
        fun analyze(frame: com.google.ar.core.Frame) {
            val det = detector ?: return

            try {
                val bmp = if (USE_STATIC_FRAME) {
                    val sb = staticBitmap
                    if (sb == null) {
                        Log.e(TAG, "Static bitmap not available")
                        return
                    }
                    sb
                } else {
                    val image = runCatching { frame.acquireCameraImage() }.getOrNull() ?: return
                    val b = cameraImageToBitmap(image)
                    image.close()
                    b
                }
                Log.d(TAG, "▶️ orig bmp size: ${bmp.width}x${bmp.height}")

                val vw = if (USE_STATIC_FRAME) overlay.width else sceneView.width
                val vh = if (USE_STATIC_FRAME) overlay.height else sceneView.height
                val viewDetections = det.detectOnView(bmp, vw, vh)
                Log.d(TAG, "Detections on view: ${viewDetections.size}")

                runOnUiThread {
                    overlay.setDetections(viewDetections)

                    val firstDet = viewDetections.firstOrNull()
                    if (firstDet == null) {
                        clearMarkers()
                        lastDetections = viewDetections
                        return@runOnUiThread
                    }

                    val prevDet = currentAnchorDetection
                    if (prevDet == null) {
                        clearMarkers()
                        placeMarker(firstDet)
                    } else {
                        val dx = firstDet.box.centerX() - prevDet.box.centerX()
                        val dy = firstDet.box.centerY() - prevDet.box.centerY()
                        val distSq = dx * dx + dy * dy
                        if (distSq > ANCHOR_MOVE_THRESHOLD_PX * ANCHOR_MOVE_THRESHOLD_PX) {
                            clearMarkers()
                            placeMarker(firstDet)
                        }
                    }
                    lastDetections = viewDetections
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error en YoloAnalyzer", e)
            }
        }
    }
}
