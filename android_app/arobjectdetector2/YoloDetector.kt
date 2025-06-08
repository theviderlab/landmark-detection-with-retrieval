package com.example.arobjectdetector2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
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
    private val net: Net,
    private val labelsAssetFile: String = "labels.txt"
) {
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
        // 1) Bitmap → Mat (RGBA)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        // 2) RGBA → BGR
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR)

        // 3) Crear blob manteniendo el tamaño original
        val blob = Dnn.blobFromImage(
            mat,
            1.0,
            Size(),           // sin redimensionar
            Scalar(0.0, 0.0, 0.0),
            /*swapRB=*/ false,
            /*crop=*/ false
        )
        net.setInput(blob)

        // 4) Ejecutar la red
        val outputs = mutableListOf<Mat>()
        net.forward(outputs, net.unconnectedOutLayersNames)
        if (outputs.size < 3) return emptyList()

        val boxes = outputs[0]
        val scores = outputs[1]
        val classes = outputs[2]

        val detections = mutableListOf<Detection>()
        val num = boxes.rows()
        for (i in 0 until num) {
            val boxArr = FloatArray(4)
            boxes.get(i, 0, boxArr)

            val scoreArr = FloatArray(1)
            scores.get(i, 0, scoreArr)

            val clsArr = FloatArray(1)
            classes.get(i, 0, clsArr)

            detections += Detection(
                cls = clsArr[0].toInt(),
                score = scoreArr[0],
                box = RectF(
                    boxArr[0],
                    boxArr[1],
                    boxArr[2],
                    boxArr[3]
                )
            )
        }

        return detections
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
