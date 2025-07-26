package com.example.arobjectdetector2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import android.os.SystemClock
import org.json.JSONObject
import kotlin.math.max

/**
 * Encapsula la ejecución del modelo YOLO exportado a ONNX.
 * El modelo contiene desde el preprocesamiento hasta el postprocesamiento,
 * por lo que sólo es necesario convertir el Bitmap a BGR y ejecutar `forward`.
 *
 * Este detector carga automáticamente el mapeo de identificadores a nombres de lugar
 * desde un archivo JSON en la carpeta de assets.
*/
class YoloDetector(
    context: Context,
    private val session: OrtSession,
    descriptorsStream: java.io.InputStream,
    private val labelMapAssetFile: String = "label_map.json"
) {
    companion object {
        private const val TAG = "YoloDetector"
    }
    // Carga el mapeo id → nombre desde assets/labelMapAssetFile
    private val classMap: Map<Int, String> =
        context.assets.open(labelMapAssetFile).use { input ->
            val text = input.bufferedReader().use { it.readText() }
            val json = JSONObject(text)
            json.keys().asSequence().associate { k ->
                k.toInt() to json.getString(k)
            }
        }

    // Tensor con la base de datos de descriptores
    private val placesTensor: OnnxTensor

    init {
        // Leer el binario de descriptores y crear el tensor (N, C+1)
        val bytes = descriptorsStream.use { it.readBytes() }
        val bb = java.nio.ByteBuffer.allocateDirect(bytes.size)
        bb.order(java.nio.ByteOrder.LITTLE_ENDIAN)
        bb.put(bytes)
        bb.rewind()

        // Las dos primeras posiciones contienen N y C+1 en formato int32
        val n = bb.int
        val c = bb.int
        Log.d(TAG, "Loaded descriptor DB: N=$n, dim=${c - 1}")
        val floatBuffer = bb.asFloatBuffer()

        placesTensor = OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            floatBuffer,
            longArrayOf(n.toLong(), c.toLong())
        )
    }

    /**
     * Ejecuta la inferencia del modelo sobre un bitmap ARGB_8888.
     * El modelo ONNX ya incluye todo el pre y postprocesado, por lo
     * que simplemente convertimos el bitmap a BGR y ejecutamos `forward`.
     */
    private fun detect(bitmap: Bitmap): List<Detection> {
        // The ONNX model expects a uint8 HWC tensor (H,W,3) in BGR order
        val width = bitmap.width
        val height = bitmap.height
        Log.d(TAG, "Running detection on ${width}x${height} bitmap")
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val buffer = java.nio.ByteBuffer.allocateDirect(width * height * 3)
        buffer.order(java.nio.ByteOrder.nativeOrder())
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            buffer.put(b.toByte())
            buffer.put(g.toByte())
            buffer.put(r.toByte())
        }
        buffer.rewind()

        val env = OrtEnvironment.getEnvironment()
        val namesIter = session.inputNames.iterator()
        val imageName = namesIter.next()
        val dbName = if (namesIter.hasNext()) namesIter.next() else "places_db"
        Log.d(TAG, "Using inputs '$imageName' and '$dbName'")
        val inputShape = longArrayOf(height.toLong(), width.toLong(), 3L)
        OnnxTensor.createTensor(env, buffer,
            inputShape,
            ai.onnxruntime.OnnxJavaType.UINT8).use { tensor ->
            Log.d(TAG, "Input tensor shape=${tensor.info.shape.contentToString()} db=${placesTensor.info.shape.contentToString()}")
            val startTime = SystemClock.uptimeMillis()
            session.run(mapOf(imageName to tensor, dbName to placesTensor)).use { result ->
                val elapsed = SystemClock.uptimeMillis() - startTime
                if (result.size() < 3) return emptyList()

                @Suppress("UNCHECKED_CAST")
                val boxes = result[0].value as Array<FloatArray>
                @Suppress("UNCHECKED_CAST")
                val scores = result[1].value as FloatArray
                @Suppress("UNCHECKED_CAST")
                val classes = result[2].value as LongArray

                Log.d(TAG, "ORT outputs -> boxes=${boxes.size}x${boxes[0].size} scores=${scores.size} classes=${classes.size} in ${elapsed}ms")

                val detections = mutableListOf<Detection>()
                for (i in scores.indices) {
                    val box = boxes[i]
                    detections += Detection(
                        cls = classes[i].toInt(),
                        score = scores[i],
                        box = RectF(box[0], box[1], box[2], box[3])
                    )
                }
                if (detections.isNotEmpty()) {
                    detections.forEachIndexed { idx, det ->
                        Log.d(
                            TAG,
                            "Det $idx -> label=${getClassName(det.cls)} score=${det.score} box=${det.box}"
                        )
                    }
                } else {
                    Log.d(TAG, "No detections")
                }
                return detections
            }
        }
    }


    /**
     * detect raw + map coords from bitmap→view
     * @param bitmap frame original
     * @param viewWidth ancho de PreviewView (o ImageView)
     * @param viewHeight alto de PreviewView (o ImageView)
     * @return lista de Detection con rects en coordenadas de vista
     */
    fun detectOnView(bitmap: Bitmap, viewWidth: Int, viewHeight: Int): List<Detection> {
        // 1) detecciones en coords de bitmap
        val raw = detect(bitmap)

        // 2) mapeo letterbox→view (igual que antes)
        val camW = bitmap.width.toFloat()
        val camH = bitmap.height.toFloat()
        val scaleV = max(viewWidth / camW, viewHeight / camH)
        val cropX = (camW * scaleV - viewWidth) / 2f
        val cropY = (camH * scaleV - viewHeight) / 2f

        return raw.map { d ->
            val l = d.box.left   * scaleV - cropX
            val t = d.box.top    * scaleV - cropY
            val r = d.box.right  * scaleV - cropX
            val b = d.box.bottom * scaleV - cropY
            Detection(
                cls   = d.cls,
                score = d.score,
                box   = RectF(
                    l.coerceAtLeast(0f),
                    t.coerceAtLeast(0f),
                    r.coerceAtMost(viewWidth.toFloat()),
                    b.coerceAtMost(viewHeight.toFloat())
                ),
                label = getClassName(d.cls)
            )
        }
    }

    /**
     * Obtiene el nombre legible de la clase a partir del índice, o el índice
     * en forma de cadena si no se encuentra en el mapa.
     */
    private fun getClassName(idx: Int): String = classMap[idx] ?: idx.toString()

    fun close() {
        Log.d(TAG, "Closing YoloDetector and releasing tensors")
        placesTensor.close()
    }
}
