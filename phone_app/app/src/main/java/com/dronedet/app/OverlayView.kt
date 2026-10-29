package com.dronedet.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import kotlin.math.max

/** One detection in the analysed (upright) camera frame's pixels. */
data class Det(val box: RectF, val score: Float)

/**
 * Draws the detections over a PreviewView in FILL_CENTER mode: the frame is scaled to cover the view
 * and centred, so a frame pixel (x, y) lands at (ox + s*x, oy + s*y).
 */
class OverlayView(context: Context) : View(context) {
    private var dets: List<Det> = emptyList()
    private var imgW = 1
    private var imgH = 1
    private val boxPaint = Paint().apply {
        color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 5f; isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.YELLOW; textSize = 42f; isAntiAlias = true; setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    fun show(d: List<Det>, w: Int, h: Int) {
        dets = d; imgW = w; imgH = h
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val s = max(width / imgW.toFloat(), height / imgH.toFloat())
        val ox = (width - imgW * s) / 2f
        val oy = (height - imgH * s) / 2f
        for (d in dets) {
            val r = RectF(ox + d.box.left * s, oy + d.box.top * s, ox + d.box.right * s, oy + d.box.bottom * s)
            // tiny far drones stay visible: at least 24 px on screen, centre kept
            if (r.width() < 24f) { val c = r.centerX(); r.left = c - 12f; r.right = c + 12f }
            if (r.height() < 24f) { val c = r.centerY(); r.top = c - 12f; r.bottom = c + 12f }
            canvas.drawRect(r, boxPaint)
            canvas.drawText("дрон %.0f%%".format(d.score * 100), r.left, r.top - 10f, textPaint)
        }
    }
}
