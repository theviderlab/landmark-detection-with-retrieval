package com.example.arobjectdetector2

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.widget.FrameLayout
import android.widget.ImageView
import android.view.WindowManager
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.arcore.createAnchorOrNull
import io.github.sceneview.loaders.ModelLoader
import io.github.sceneview.node.ModelNode
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.view.View

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
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
        // Initialize the loader that will create 3-D model instances
        modelLoader = ModelLoader(sceneView.engine, this)
        cameraExecutor = Executors.newSingleThreadExecutor()

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
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                startCameraX()
            } else {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    CAMERA_PERMISSION_REQUEST
                )
            }
        }

        loadDnnModel()

        previewView.post {
            // espera a que previewView (y overlay) tengan width/height válidos
            detector?.let { det ->
                staticBitmap?.let { bmp ->
                    val vw = if (USE_STATIC_FRAME) overlay.width else previewView.width
                    val vh = if (USE_STATIC_FRAME) overlay.height else previewView.height
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

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                startCameraX()
            } else {
                Toast.makeText(this, "Necesito permiso de cámara para funcionar", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun startCameraX() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, YoloAnalyzer())
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this,
                CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
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
        val centerX = det.box.centerX()
        val centerY = det.box.centerY()
        val hit = frame.hitTest(centerX, centerY).firstOrNull() ?: return

        // Remove previous anchor if present
        currentAnchorNode?.let { node ->
            sceneView.removeChildNode(node)
            node.destroy()
            currentAnchorNode = null
        }

        val anchor = hit.createAnchorOrNull() ?: return
        val anchorNode = io.github.sceneview.ar.node.AnchorNode(sceneView.engine, anchor)

        // Load the marker model and attach it to the new anchor
        val modelInstance = modelLoader.createModelInstance("location.fbx")
        val modelNode = ModelNode(modelInstance)
        anchorNode.addChildNode(modelNode)
        sceneView.addChildNode(anchorNode)
        currentAnchorNode = anchorNode
        currentAnchorDetection = det
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
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
        val rot = image.imageInfo.rotationDegrees
        return if (rot != 0) {
            val m = Matrix().apply { postRotate(rot.toFloat()) }
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        } else {
            bmp
        }
    }


    private inner class YoloAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            val det = detector
            if (det == null) {
                imageProxy.close()
                return
            }

            try {
                // 1) Frame → Bitmap
                val bmp = if (USE_STATIC_FRAME) {
                    val sb = staticBitmap
                    if (sb == null) {
                        Log.e(TAG, "Static bitmap not available")
                        imageProxy.close()
                        return
                    }
                    sb
                } else {
                    imageProxyToBitmap(imageProxy)
                }
                Log.d(TAG, "▶️ orig bmp size: ${bmp.width}x${bmp.height}")

                val vw = if (USE_STATIC_FRAME) overlay.width else previewView.width
                val vh = if (USE_STATIC_FRAME) overlay.height else previewView.height
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
            } finally {
                imageProxy.close()
            }
        }
    }
}
