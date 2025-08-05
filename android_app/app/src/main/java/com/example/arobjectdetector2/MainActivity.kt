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
import android.widget.TextView
import android.view.WindowManager
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.loaders.ModelLoader
import io.github.sceneview.node.ModelNode
import io.github.sceneview.node.ViewNode2
import io.github.sceneview.node.ViewNodeWindowManager
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import com.google.ar.core.Config
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.view.View
import android.opengl.Matrix
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val USE_STATIC_FRAME = false
        /** Pixel distance threshold to recreate the AR anchor. */
        private const val ANCHOR_MOVE_THRESHOLD_PX = 50f

        // Cache the descriptor database in memory so it is loaded only once
        private var descriptorBytes: ByteArray? = null

        /** When true, show the detection BoxOverlay for debugging purposes. */
        private const val SHOW_BOX_OVERLAY = false
        /** Minimum time between detection runs. */
        private const val DETECTION_INTERVAL_MS = 1000L

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

    /** Placed markers indexed by detection ID. */
    private val markers = mutableMapOf<Int, io.github.sceneview.ar.node.AnchorNode>()
    private val lastDetectionsMap = mutableMapOf<Int, Detection>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlay     = findViewById(R.id.overlay)
        overlay.visibility = if (SHOW_BOX_OVERLAY) View.VISIBLE else View.GONE
        sceneView   = findViewById(R.id.sceneView)
        sceneView.lifecycle = lifecycle
        sceneView.viewNodeWindowManager = ViewNodeWindowManager(this)
        sceneView.session?.let { session ->
            val config = Config(session).apply {
                // Enable plane detection so hitTest can return reliable poses
                planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
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
                        if (SHOW_BOX_OVERLAY) {
                            overlay.setDetections(viewDets)
                        }
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
        markers.values.forEach { node ->
            sceneView.removeChildNode(node)
            node.destroy()
        }
        markers.clear()
        lastDetectionsMap.clear()
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

        // Read depth at the detection center in screen coordinates
        val depthMeters = runCatching { frame.acquireDepthImage16Bits() }.getOrNull()?.use { img ->
            val depthW = img.width
            val depthH = img.height
            val dx = ((centerX / sceneView.width) * depthW).toInt().coerceIn(0, depthW - 1)
            val dy = ((centerY / sceneView.height) * depthH).toInt().coerceIn(0, depthH - 1)
            val plane = img.planes[0]
            val rowStride = plane.rowStride
            val pxStride = plane.pixelStride
            val buffer = plane.buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
            val index = dy * rowStride + dx * pxStride
            if (index >= 0 && index <= buffer.capacity() - 2) {
                buffer.position(index)
                (buffer.short.toInt() and 0xFFFF) / 1000f
            } else {
                Float.NaN
            }
        } ?: Float.NaN

        // Try to obtain a safe pose from hitTest (planes or depth)
        val hitPose = frame.hitTest(centerX, centerY).firstOrNull()?.hitPose

        val pose = if (hitPose != null) {
            hitPose
        } else {
            if (!depthMeters.isFinite() || depthMeters <= 0f) {
                Log.w(TAG, "Cannot place marker: invalid depth and no hit result")
                return
            }

            // Convert 2-D screen point to a 3-D coordinate using view/projection matrices
            val projMatrix = FloatArray(16)
            frame.camera.getProjectionMatrix(projMatrix, 0, 0.1f, 100f)
            val viewMatrix = FloatArray(16)
            frame.camera.getViewMatrix(viewMatrix, 0)
            val viewProj = FloatArray(16)
            android.opengl.Matrix.multiplyMM(viewProj, 0, projMatrix, 0, viewMatrix, 0)
            val invViewProj = FloatArray(16)
            android.opengl.Matrix.invertM(invViewProj, 0, viewProj, 0)
            val ndcX = 2f * centerX / sceneView.width - 1f
            val ndcY = 1f - 2f * centerY / sceneView.height
            val clip = floatArrayOf(ndcX, ndcY, -1f, 1f)
            val out = FloatArray(4)
            android.opengl.Matrix.multiplyMV(out, 0, invViewProj, 0, clip, 0)
            for (i in 0 until 3) out[i] /= out[3]
            val dir = floatArrayOf(out[0], out[1], out[2])
            val norm = kotlin.math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            dir[0] /= norm
            dir[1] /= norm
            dir[2] /= norm

            val translation = floatArrayOf(
                dir[0] * depthMeters,
                dir[1] * depthMeters,
                dir[2] * depthMeters
            )
            frame.camera.pose.compose(Pose.makeTranslation(translation))
        }

        val anchor = sceneView.session?.createAnchor(pose) ?: return

        // Remove previous anchor for this ID if present
        markers[det.cls]?.let { node ->
            sceneView.removeChildNode(node)
            node.destroy()
        }

        val anchorNode = io.github.sceneview.ar.node.AnchorNode(sceneView.engine, anchor)

        // Load the marker model and attach it to the new anchor
        val modelInstance = modelLoader.createModelInstance("location.glb")
        val modelNode = ModelNode(modelInstance)
        anchorNode.addChildNode(modelNode)

        // Create a child node displaying the detection label
        val windowManager = sceneView.viewNodeWindowManager ?: run {
            Log.w(TAG, "viewNodeWindowManager is null; marker not placed")
            return
        }
        val textNode = ViewNode2(
            sceneView.engine,
            windowManager,
            sceneView.materialLoader,
            R.layout.label_renderable
        )
        textNode.layout.getChildAt(0)
            .findViewById<TextView>(R.id.labelText).text = det.label
        textNode.position = dev.romainguy.kotlin.math.Float3(0f, -0.1f, 0f)
        anchorNode.addChildNode(textNode)

        sceneView.addChildNode(anchorNode)
        markers[det.cls] = anchorNode
        lastDetectionsMap[det.cls] = det
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


    private inner class YoloAnalyzer {
        private var lastDetectionTime = 0L
        private var lastViewDetections: List<Detection> = emptyList()
        private var lastFrameTimestamp = 0L

        fun analyze(frame: com.google.ar.core.Frame) {
            val det = detector ?: return

            val frameTs = frame.timestamp
            if (frameTs <= lastFrameTimestamp) {
                return
            }
            lastFrameTimestamp = frameTs

            val now = android.os.SystemClock.uptimeMillis()
            val shouldDetect = now - lastDetectionTime >= DETECTION_INTERVAL_MS


            try {
                val viewDetections = if (shouldDetect) {
                    val bmp = if (USE_STATIC_FRAME) {
                        val sb = staticBitmap
                        if (sb == null) {
                            Log.e(TAG, "Static bitmap not available")
                            return
                        }
                        sb
                    } else {
                        val image = runCatching { frame.acquireCameraImage() }.getOrNull() ?: return
                        val b = image.use { cameraImageToBitmap(it) }
                        b
                    }
                    Log.d(TAG, "▶️ orig bmp size: ${bmp.width}x${bmp.height}")

                    val vw = if (USE_STATIC_FRAME) overlay.width else sceneView.width
                    val vh = if (USE_STATIC_FRAME) overlay.height else sceneView.height
                    val detections = det.detectOnView(bmp, vw, vh)
                    lastDetectionTime = now
                    lastViewDetections = detections
                    Log.d(TAG, "Detections on view: ${detections.size}")
                    detections
                } else {
                    lastViewDetections
                }

                runOnUiThread {
                    if (SHOW_BOX_OVERLAY) {
                        overlay.setDetections(viewDetections)
                    }

                    val currentIds = mutableSetOf<Int>()
                    for (d in viewDetections) {
                        currentIds.add(d.cls)
                        val prevDet = lastDetectionsMap[d.cls]
                        if (sceneView.viewNodeWindowManager == null) {
                            Log.w(TAG, "viewNodeWindowManager is null; skipping marker placement")
                            continue
                        }
                        if (prevDet == null) {
                            placeMarker(d)
                        } else {
                            val dx = d.box.centerX() - prevDet.box.centerX()
                            val dy = d.box.centerY() - prevDet.box.centerY()
                            val distSq = dx * dx + dy * dy
                            if (distSq > ANCHOR_MOVE_THRESHOLD_PX * ANCHOR_MOVE_THRESHOLD_PX) {
                                placeMarker(d)
                            } else {
                                lastDetectionsMap[d.cls] = d
                            }
                        }
                    }

                    val toRemove = markers.keys - currentIds
                    for (id in toRemove) {
                        markers[id]?.let { node ->
                            sceneView.removeChildNode(node)
                            node.destroy()
                        }
                        markers.remove(id)
                        lastDetectionsMap.remove(id)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error en YoloAnalyzer", e)
            }
        }
    }
}
