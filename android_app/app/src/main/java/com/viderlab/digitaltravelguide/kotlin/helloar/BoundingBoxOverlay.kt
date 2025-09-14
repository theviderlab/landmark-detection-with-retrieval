package com.viderlab.digitaltravelguide.kotlin.helloar

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

class BoundingBoxOverlay(context: Context) : View(context) {
    private var bboxes: List<LabeledBoundingBox>? = null
  private val bboxPaint = Paint().apply {
    style = Paint.Style.STROKE
    color = Color.RED
    strokeWidth = 4f
  }
  private val textPaint = Paint().apply {
    style = Paint.Style.FILL
    color = Color.RED
    textSize = 40f
  }

  fun showBoundingBoxes(bboxes: List<LabeledBoundingBox>) {
    // Bounding boxes are provided in view coordinates. Simply draw them on the canvas.
    this.bboxes = bboxes
    invalidate()
  }

  fun clearBoundingBoxes() {
    this.bboxes = null
    invalidate()
  }

    override fun onDraw(canvas: Canvas) {
      super.onDraw(canvas)
      bboxes?.forEach { (bbox, label, score) ->
        canvas.drawRect(bbox.x1, bbox.y1, bbox.x2, bbox.y2, bboxPaint)
        val padding = 8f
        val labelX = bbox.x1 + padding
        val labelY = bbox.y1 + padding - textPaint.ascent()
        canvas.drawText(label, labelX, labelY, textPaint)
        val scoreY = labelY + textPaint.textSize + padding
        canvas.drawText("%.2f".format(score), labelX, scoreY, textPaint)
      }
    }
  }
