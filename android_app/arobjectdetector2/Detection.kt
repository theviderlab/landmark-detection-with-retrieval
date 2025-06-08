package com.example.arobjectdetector2

import android.graphics.RectF

data class Detection(
    val cls: Int,
    val score: Float,
    val box: RectF,
    val label: String = cls.toString()    // opción por defecto
)

fun iou(a: RectF, b: RectF): Float {
    val left   = maxOf(a.left, b.left)
    val top    = maxOf(a.top, b.top)
    val right  = minOf(a.right, b.right)
    val bottom = minOf(a.bottom, b.bottom)
    val w = maxOf(0f, right - left)
    val h = maxOf(0f, bottom - top)
    val inter = w * h
    val areaA = (a.right - a.left) * (a.bottom - a.top)
    val areaB = (b.right - b.left) * (b.bottom - b.top)
    return inter / (areaA + areaB - inter)
}

fun nms(dets: List<Detection>, iouThresh: Float): List<Detection> {
    val out = mutableListOf<Detection>()
    val list = dets.sortedByDescending { it.score }.toMutableList()
    while (list.isNotEmpty()) {
        val best = list.removeAt(0)
        out += best
        val it = list.iterator()
        while (it.hasNext()) {
            if (iou(best.box, it.next().box) > iouThresh) it.remove()
        }
    }
    return out
}
