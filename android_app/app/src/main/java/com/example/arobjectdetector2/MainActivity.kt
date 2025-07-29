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

        // Read depth at the detection center in screen coordinates
        val depthImage = runCatching { frame.acquireDepthImage() }.getOrNull()
        val depthMeters = depthImage?.let { img ->
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
        depthImage?.close()

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

        val depth = if (depthMeters.isNaN() || depthMeters <= 0f) 2f else depthMeters
        val translation = floatArrayOf(dir[0] * depth, dir[1] * depth, dir[2] * depth)
        val pose = frame.camera.pose.compose(Pose.makeTranslation(translation))
        val anchor = sceneView.session?.createAnchor(pose) ?: return

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
