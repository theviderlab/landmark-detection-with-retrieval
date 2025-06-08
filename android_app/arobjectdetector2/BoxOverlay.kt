package com.example.arobjectdetector2

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class BoxOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        color = 0xffff0000.toInt()      // rojo
    }
    private val textPaint = Paint().apply {
        color = 0xffff0000.toInt()
        textSize = 48f
    }
    private var detections: List<Detection> = emptyList()

    fun setDetections(detections: List<Detection>) {
        this.detections = detections
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (det in detections) {
            val r = det.box
            canvas.drawRect(r, boxPaint)
            canvas.drawText(
                "${det.label}: ${"%.2f".format(det.score)}",
                r.left,
                r.top - 10,
                textPaint
            )
        }
    }
}
