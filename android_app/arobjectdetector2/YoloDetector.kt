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
import kotlin.math.min

/**
 * Encapsula la lógica de preprocesado, forward, postprocesado,
 * supresión de no-máximos (NMS) y mapeo de índices a nombres de clase
 * para un modelo YOLO exportado a ONNX.
 *
 * Este detector carga automáticamente los nombres de clase desde un archivo de labels en assets.
 */
class YoloDetector(
    context: Context,
    private val net: Net,
    private val inputSize: Size = Size(640.0, 640.0),
    private val confThreshold: Float = 0.5f,
    private val nmsThreshold: Float = 0.45f,
    private val labelsAssetFile: String = "labels.txt"
) {
    // Carga nombres de clase desde assets/labelsAssetFile
    private val classNames: List<String> = context.assets.open(labelsAssetFile)
        .bufferedReader()
        .useLines { it.toList() }

    /**
     * Ejecuta detección sobre un bitmap ARGB_8888 de tamaño arbitrario.
     * Devuelve detecciones post-NMS con coordenadas en píxeles del bitmap.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        // 1) Bitmap → Mat (RGBA)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        // 2) RGBA → BGR
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR)

        // 3) Letterbox params
        val origW = bitmap.width.toDouble()
        val origH = bitmap.height.toDouble()
        val scale = min(inputSize.width / origW, inputSize.height / origH)
        val padW = (inputSize.width - origW * scale) / 2.0
        val padH = (inputSize.height - origH * scale) / 2.0

        // 4) Resize al tamaño de input
        Imgproc.resize(mat, mat, inputSize)

        // 5) Blob + forward
        val blob = Dnn.blobFromImage(
            mat,
            1.0,
            inputSize,
            Scalar(0.0, 0.0, 0.0),
            /*swapRB=*/ true,
            /*crop=*/ false
        )
        net.setInput(blob)

        // 6) Obtener salida
        val outputs = mutableListOf<Mat>()
        net.forward(outputs, net.unconnectedOutLayersNames)
        val output3d = outputs[0]

        // 7) Reformar a 2D rows=boxes cols=attributes
        val numBoxes = output3d.size(2).toInt()
        val mat2d = output3d.reshape(1, numBoxes)

        // 8) Parse detecciones crudas
        val raw = mutableListOf<Detection>()
        for (i in 0 until mat2d.rows()) {
            val row = FloatArray(mat2d.cols())
            mat2d.get(i, 0, row)

            // Formato YOLOv8: [cx, cy, w, h, obj_conf, cls1, cls2, ...]
            val cxNorm = row[0]
            val cyNorm = row[1]
            val wNorm = row[2]
            val hNorm = row[3]

            // Mejor clase
            val classScores = row.sliceArray(4 until row.size)
            val bestClass = classScores.indices.maxByOrNull { classScores[it] } ?: -1
            val clsC = if (bestClass >= 0) classScores[bestClass] else 0f

            val conf = clsC
            if (conf > confThreshold) {
                // Coordenadas en inputSize
                val cxIn = cxNorm * inputSize.width
                val cyIn = cyNorm * inputSize.height
                val wIn = wNorm * inputSize.width
                val hIn = hNorm * inputSize.height

                // Revertir letterbox+scale → coords en bitmap
                val left = ((cxIn - wIn / 2 - padW) / scale).toFloat()
                val top = ((cyIn - hIn / 2 - padH) / scale).toFloat()
                val right = ((cxIn + wIn / 2 - padW) / scale).toFloat()
                val bottom = ((cyIn + hIn / 2 - padH) / scale).toFloat()

                raw += Detection(
                    cls = bestClass,
                    score = conf,
                    box = RectF(
                        left.coerceAtLeast(0f),
                        top.coerceAtLeast(0f),
                        right.coerceAtMost(bitmap.width.toFloat()),
                        bottom.coerceAtMost(bitmap.height.toFloat())
                    )
                )
            }
        }

        // 9) Aplicar NMS y devolver detecciones finales
        return applyNMS(raw, nmsThreshold)
    }

    /**
     * Supresión de no-máximos: filtra detecciones solapadas.
     */
    private fun applyNMS(detections: List<Detection>, iouThresh: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val kept = mutableListOf<Detection>()
        for (det in sorted) {
            if (kept.none { iou(it.box, det.box) > iouThresh }) {
                kept += det
            }
        }
        return kept
    }

    /**
     * Calcula el IoU entre dos RectF.
     */
    private fun iou(a: RectF, b: RectF): Float {
        val x1 = max(a.left, b.left)
        val y1 = max(a.top, b.top)
        val x2 = min(a.right, b.right)
        val y2 = min(a.bottom, b.bottom)
        val interW = (x2 - x1).coerceAtLeast(0f)
        val interH = (y2 - y1).coerceAtLeast(0f)
        val interArea = interW * interH
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return interArea / (areaA + areaB - interArea)
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
