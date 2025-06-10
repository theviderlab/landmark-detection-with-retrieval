package com.example.arobjectdetector2

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.DisplayMetrics
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
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.view.View

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
        private const val USE_STATIC_FRAME = true

        init {
            System.loadLibrary("opencv_java4")
            Log.d(TAG, "OpenCV native lib loaded")
        }
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: BoxOverlay
    private lateinit var cameraExecutor: ExecutorService
    private var dnnNet: Net? = null
    private var detector: YoloDetector? = null  // ← Nuevo
    private var staticBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlay     = findViewById(R.id.overlay)
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
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    CAMERA_PERMISSION_REQUEST
                )
            }
        }

        loadDnnModel()


        val metrics = DisplayMetrics().also { windowManager.defaultDisplay.getMetrics(it) }
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
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

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
                val onnxName = "pipeline-yolo-cvnet-sg-v1.onnx"
                val tmpFile = File(cacheDir, onnxName)
                assets.open(onnxName).use { input ->
                    tmpFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                dnnNet = Dnn.readNetFromONNX(tmpFile.absolutePath)
                Log.d(TAG, "DNN ONNX cargado correctamente")

                // Inicializa el detector con la red y el tamaño de entrada
                detector = YoloDetector(
                    this,
                    net = dnnNet!!
                )
                onDetectorReady()
            } catch (e: Exception) {
                Log.e(TAG, "Error cargando el modelo ONNX", e)
            }
        }.start()
    }

    private fun onDetectorReady() {
        runOnUiThread {
            if (USE_STATIC_FRAME) {
                val det = detector ?: return@runOnUiThread
                val bmp = staticBitmap ?: return@runOnUiThread
                val viewDets = det.detectOnView(
                    bmp,
                    previewView.width,
                    previewView.height
                )
                overlay.setDetections(viewDets)
            } else {
                if (ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.CAMERA
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    startCameraX()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
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

        val yuvImage = YuvImage(nv21, ImageFormat.NV21,
            image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height), 90, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
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

                val viewDetections = det.detectOnView(bmp, previewView.width, previewView.height)

                runOnUiThread {
                    overlay.setDetections(viewDetections)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error en YoloAnalyzer", e)
            } finally {
                imageProxy.close()
            }
        }
    }
}
