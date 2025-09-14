package com.viderlab.digitaltravelguide.kotlin.helloar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

/**
 * Simple overlay view that renders a list of detections in the bottom-left corner.
 * Each line contains the detection label and score on a semi-transparent background.
 */
class DetectionListOverlay(context: Context) : View(context) {
    private var detections: List<LabeledBoundingBox>? = null

  private val textPaint = Paint().apply {
    style = Paint.Style.FILL
    color = Color.WHITE
    textSize = 24f
  }

  private val backgroundPaint = Paint().apply {
    style = Paint.Style.FILL
    color = Color.BLACK
    alpha = 160
  }

  /**
   * Displays a list of detections. Passing an empty list hides the overlay.
   */
  fun showDetections(detections: List<LabeledBoundingBox>) {
    this.detections = if (detections.isEmpty()) null else detections
    invalidate()
  }

  /** Clears the detection list overlay. */
  fun clearDetections() {
    detections = null
    invalidate()
  }

    override fun onDraw(canvas: Canvas) {
      super.onDraw(canvas)
      val list = detections ?: return

    val lines = list.map { "${it.label}: %.2f".format(it.score) }
    val padding = 8f
    val lineHeight = -textPaint.ascent() + textPaint.descent()
    val bottom = height - padding
    val top = bottom - lineHeight * lines.size - padding * 2

    canvas.drawRect(0f, top, width.toFloat(), bottom, backgroundPaint)

      var y = top + padding - textPaint.ascent()
      for (line in lines) {
        canvas.drawText(line, padding, y, textPaint)
        y += lineHeight
      }
    }
  }
