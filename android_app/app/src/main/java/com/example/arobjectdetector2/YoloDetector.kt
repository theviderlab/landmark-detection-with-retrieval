package com.example.arobjectdetector2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import kotlin.math.max

/**
 * Encapsula la ejecución del modelo YOLO exportado a ONNX.
 * El modelo contiene desde el preprocesamiento hasta el postprocesamiento,
 * por lo que sólo es necesario convertir el Bitmap a BGR y ejecutar `forward`.
 *
 * Este detector carga automáticamente los nombres de clase desde un archivo de labels en assets.
 */
class YoloDetector(
    context: Context,
    private val session: OrtSession,
    private val labelsAssetFile: String = "labels.txt"
) {
    companion object {
        private const val TAG = "YoloDetector"
    }
    // Carga nombres de clase desde assets/labelsAssetFile
    private val classNames: List<String> = context.assets.open(labelsAssetFile)
        .bufferedReader()
        .useLines { it.toList() }

    /**
     * Ejecuta la inferencia del modelo sobre un bitmap ARGB_8888.
     * El modelo ONNX ya incluye todo el pre y postprocesado, por lo
     * que simplemente convertimos el bitmap a BGR y ejecutamos `forward`.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        // The ONNX model expects a uint8 HWC tensor (H,W,3) in BGR order
        val width = bitmap.width
        val height = bitmap.height
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
        val inputName = session.inputNames.iterator().next()
        Log.d(TAG, "Using input '$inputName' -> shape ($height, $width, 3)")
        OnnxTensor.createTensor(env, buffer,
            longArrayOf(height.toLong(), width.toLong(), 3L),
            ai.onnxruntime.OnnxJavaType.UINT8).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                if (result.size() < 3) return emptyList()

                val boxes = result[0].value as Array<FloatArray>
                val scores = result[1].value as FloatArray
                val classes = result[2].value as LongArray

                Log.d(TAG, "ORT outputs -> boxes=${boxes.size} scores=${scores.size}")

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
                    val d = detections.first()
                    Log.d(TAG, "First det -> cls=${d.cls} score=${d.score} box=${d.box}")
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
     * Obtiene el nombre de clase a partir del índice, o el índice si no existe.
     */
    fun getClassName(idx: Int): String = classNames.getOrNull(idx) ?: idx.toString()
}
