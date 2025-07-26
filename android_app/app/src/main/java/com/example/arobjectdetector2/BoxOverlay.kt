package com.example.arobjectdetector2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.drawable.BitmapDrawable
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.example.arobjectdetector2.R

private fun Context.drawableToBitmap(resId: Int): Bitmap {
    val drawable = ContextCompat.getDrawable(this, resId) ?: return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
    if (drawable is BitmapDrawable) {
        return drawable.bitmap
    }
    val bitmap = Bitmap.createBitmap(
        drawable.intrinsicWidth,
        drawable.intrinsicHeight,
        Bitmap.Config.ARGB_8888
    )
    val canvas = Canvas(bitmap)
    drawable.setBounds(0, 0, canvas.width, canvas.height)
    drawable.draw(canvas)
    return bitmap
}

class BoxOverlay @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0
) : View(context, attrs, defStyle) {

    companion object {
        /** When true, draw a fixed debug rectangle regardless of detections. */
        const val SHOW_DEBUG_BOX = false
    }

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        color = 0xffff0000.toInt()      // rojo
    }
    private val textPaint = Paint().apply {
        color = 0xffff0000.toInt()
        textSize = 48f
    }
    private val locationBitmap =
        context.drawableToBitmap(R.drawable.ic_location_3d).let {
            Bitmap.createScaledBitmap(it, it.width * 2, it.height * 2, true)
        }
    /** Reusable rectangle to avoid allocations during drawing. */
    private val debugRect = RectF()
    private var detections: List<Detection> = emptyList()

    fun setDetections(detections: List<Detection>) {
        this.detections = detections
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (det in detections) {
            val r = det.box
            // canvas.drawRect(r, boxPaint)

            val centerX = r.centerX()
            val iconLeft = centerX - locationBitmap.width / 2f
            val iconTop = r.top - locationBitmap.height
            canvas.drawBitmap(locationBitmap, iconLeft, iconTop, null)

            val text = det.label
            val overlayWidth = width.toFloat()
            val textY = iconTop + locationBitmap.height + textPaint.textSize

            var remaining = text
            var lineIndex = 0
            while (remaining.isNotEmpty()) {
                val count = textPaint.breakText(remaining, true, overlayWidth, null)
                val line = remaining.substring(0, count)
                val lineWidth = textPaint.measureText(line)
                val textX = centerX - lineWidth / 2f
                val lineY = textY + lineIndex * textPaint.fontSpacing
                canvas.drawText(line, textX, lineY, textPaint)
                remaining = remaining.substring(count)
                lineIndex += 1
            }
        }
        if (SHOW_DEBUG_BOX) {
            debugRect.set(
                width * 0.2f,
                height * 0.2f,
                width * 0.8f,
                height * 0.8f,
            )
            canvas.drawRect(debugRect, boxPaint)
            canvas.drawText("debug", debugRect.left, debugRect.top - 10, textPaint)
        }
    }
    override fun gatherTransparentRegion(region: android.graphics.Region?): Boolean {
        // Disable transparent region hints to avoid surfaceflinger warnings.
        return false
    }
}
