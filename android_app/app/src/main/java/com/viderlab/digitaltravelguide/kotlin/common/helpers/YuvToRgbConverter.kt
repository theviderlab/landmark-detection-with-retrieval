/*
 * Copyright 2024
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.viderlab.digitaltravelguide.kotlin.common.helpers

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.renderscript.Type

/**
 * Helper class that converts a [Image] in YUV_420_888 format into an RGB [Bitmap].
 */
class YuvToRgbConverter(context: Context) {
  private val rs = RenderScript.create(context)
  private val script = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
  private var yuvBuffer: ByteArray? = null
  private var input: Allocation? = null
  private var output: Allocation? = null

  fun yuvToRgb(image: Image): Bitmap {
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    if (yuvBuffer == null || yuvBuffer!!.size != ySize + uSize + vSize) {
      yuvBuffer = ByteArray(ySize + uSize + vSize)
    }
    val bytes = yuvBuffer!!
    yBuffer.get(bytes, 0, ySize)
    vBuffer.get(bytes, ySize, vSize)
    uBuffer.get(bytes, ySize + vSize, uSize)

    if (input == null || input!!.type.x != bytes.size) {
      val yuvType = Type.Builder(rs, Element.U8(rs)).setX(bytes.size)
      input = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)
    }
    if (output == null || output!!.type.x != image.width || output!!.type.y != image.height) {
      val rgbaType = Type.Builder(rs, Element.RGBA_8888(rs)).setX(image.width).setY(image.height)
      output = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT)
    }

    input!!.copyFrom(bytes)
    script.setInput(input)
    script.forEach(output)

    val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
    output!!.copyTo(bitmap)
    return bitmap
  }
}
