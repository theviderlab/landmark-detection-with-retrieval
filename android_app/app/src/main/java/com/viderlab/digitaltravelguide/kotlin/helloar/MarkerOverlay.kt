package com.viderlab.digitaltravelguide.kotlin.helloar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

class MarkerOverlay(context: Context) : View(context) {
  data class Marker(val x: Float, val y: Float, val label: String)

    private var markers: List<Marker>? = null
    private var drawBoxes = false

  private val textPaint = Paint().apply {
    style = Paint.Style.FILL
    color = Color.WHITE
    textSize = 48f
  }

  private val backgroundPaint = Paint().apply {
    style = Paint.Style.FILL
    color = Color.BLACK
    alpha = 160
  }

  private val boxPaint = Paint().apply {
    style = Paint.Style.STROKE
    color = Color.RED
    strokeWidth = 2f
  }

  fun updateAnchors(markers: List<Marker>, drawBoxes: Boolean = false) {
    this.markers = if (markers.isEmpty()) null else markers
    this.drawBoxes = drawBoxes
    invalidate()
  }

    override fun onDraw(canvas: Canvas) {
      super.onDraw(canvas)
      markers?.forEach { marker ->
        val textWidth = textPaint.measureText(marker.label)
        val textHeight = -textPaint.ascent() + textPaint.descent()
      val left = marker.x
      val top = marker.y + 8f
      val right = left + textWidth
      val bottom = top + textHeight
      canvas.drawRect(left, top, right, bottom, backgroundPaint)
        val baseline = bottom - textPaint.descent()
        canvas.drawText(marker.label, left, baseline, textPaint)
        if (drawBoxes) {
          canvas.drawRect(left, top, right, bottom, boxPaint)
        }
      }
    }
  }
