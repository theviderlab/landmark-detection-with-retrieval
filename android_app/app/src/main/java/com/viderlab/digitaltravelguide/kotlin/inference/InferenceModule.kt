package com.viderlab.digitaltravelguide.kotlin.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.text.Charsets
import kotlin.math.min
import androidx.core.graphics.get

/**
 * Loads an ONNX model from assets and provides image inference utilities.
 */
class InferenceModule(private val context: Context, config: PipelineConfig) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private var session: OrtSession? = null
    private var placesDbTensor: OnnxTensor? = null
    private var labelMap: Map<Int, String> = emptyMap()

    private var modelPath: String = config.model
    private var embeddingsPath: String = config.embeddings
    private var labelMapPath: String? = config.labelMap

    init {
        load(config)
    }

    fun load(config: PipelineConfig) {
        session?.close()
        session = null
        placesDbTensor?.close()
        placesDbTensor = null
        labelMap = emptyMap()

        modelPath = config.model
        embeddingsPath = config.embeddings
        labelMapPath = config.labelMap

        session = try {
            val model = context.assets.open(modelPath).use { it.readBytes() }
            env.createSession(model).also { Log.d(TAG, "ORT session initialized") }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ORT session", e)
            null
        }

        placesDbTensor = try {
            val bytes = context.assets.open(embeddingsPath).use { it.readBytes() }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val rows = buffer.int
            val cols = buffer.int
            val expected = rows * cols
            val floatBuffer = buffer.asFloatBuffer()
            if (floatBuffer.remaining() != expected) {
                throw IllegalStateException("Invalid data length: expected $expected but was ${floatBuffer.remaining()}")
            }
            val floats = FloatArray(expected)
            floatBuffer.get(floats)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), longArrayOf(rows.toLong(), cols.toLong())).also {
                Log.d(TAG, "placesDbTensor initialized with shape [$rows, $cols]")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize placesDbTensor", e)
            null
        }

        labelMap = try {
            val path = labelMapPath
            if (path != null) {
                context.assets.open(path).use { stream ->
                    val json = JSONObject(stream.reader(Charsets.UTF_8).readText())
                    val map = mutableMapOf<Int, String>()
                    val keys = json.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        map[key.toInt()] = json.getString(key)
                    }
                    map
                }
            } else {
                emptyMap()
            }
        } catch (e: Exception) {
            emptyMap()
        }
    }

    /**
     * Runs inference on [bitmap] and returns a list of [Detection] results.
     *
     * The model outputs bounding boxes as absolute pixel coordinates in
     * `[left, top, right, bottom]` order with the origin at the top-left of the
     * input [bitmap]. Coordinates are already scaled to the size of the input
     * image, so this method simply clamps them to `[0, width]` and `[0, height]`
     * without performing any additional scaling.
     */
    fun runInference(bitmap: Bitmap): List<Detection> {
        val ortSession = session ?: run {
            Log.w(TAG, "ORT session is null; returning empty list")
            return emptyList()
        }
        val embeddings = placesDbTensor ?: run {
            Log.w(TAG, "placesDbTensor is null; returning empty list")
            return emptyList()
        }

        val width: Int = bitmap.width
        val height: Int = bitmap.height
        val buffer = ByteBuffer.allocateDirect(3 * width * height)
            .order(ByteOrder.nativeOrder())
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap[x, y]
                buffer.put((pixel and 0xFF).toByte())
                buffer.put(((pixel shr 8) and 0xFF).toByte())
                buffer.put(((pixel shr 16) and 0xFF).toByte())
            }
        }
        buffer.rewind()

        val inputNames = ortSession.inputNames
        if (!inputNames.contains("image_bgr")) {
            Log.w(TAG, "\"image_bgr\" not found in input names: $inputNames")
            return emptyList()
        }
        val embeddingInput = inputNames.firstOrNull { it != "image_bgr" } ?: run {
            Log.w(TAG, "Embedding input not found; returning empty list")
            return emptyList()
        }
        Log.d(TAG, "Input tensor dims: [height=$height, width=$width, channels=3]")

        OnnxTensor.createTensor(
            env,
            buffer,
            longArrayOf(height.toLong(), width.toLong(), 3),
            OnnxJavaType.UINT8
        ).use { tensor ->
            ortSession.run(
                mapOf(
                    "image_bgr" to tensor,
                    embeddingInput to embeddings
                )
            ).use { result ->
                val boxes = result[0].value as? Array<FloatArray> ?: run {
                    Log.w(TAG, "Boxes output missing or invalid; returning empty list")
                    return emptyList()
                }
                val scores = result[1].value as? FloatArray ?: run {
                    Log.w(TAG, "Scores output missing or invalid; returning empty list")
                    return emptyList()
                }
                val classes = result[2].value as? LongArray ?: run {
                    Log.w(TAG, "Classes output missing or invalid; returning empty list")
                    return emptyList()
                }

                Log.d(TAG, "boxes count=${boxes.size}, classes count=${classes.size}")
                boxes.take(3).forEachIndexed { index, b ->
                    Log.d(TAG, "box[$index]=${b.joinToString()}")
                }
                classes.take(3).forEachIndexed { index, c ->
                    Log.d(TAG, "class[$index]=$c")
                }

                val count = min(min(boxes.size, classes.size), scores.size)
                val detections = mutableListOf<Detection>()
                for (i in 0 until count) {
                    val b = boxes[i]
                    // The model already provides absolute pixel coordinates for the
                    // bounding box, so we just clamp them to the image bounds to avoid
                    // any out-of-range values.
                    val left = b[0].toInt().coerceIn(0, width)
                    val top = b[1].toInt().coerceIn(0, height)
                    val right = b[2].toInt().coerceIn(0, width)
                    val bottom = b[3].toInt().coerceIn(0, height)
                    val rect = Rect(left, top, right, bottom)
                    val label = labelMap[classes[i].toInt()] ?: classes[i].toInt().toString()
                    val score = scores[i]
                    Log.d(TAG, "detection[$i]: label=$label score=$score raw=${b.joinToString()} rect=$rect")
                    detections.add(Detection(rect, label, score))
                }
                return detections
            }
        }
    }

    companion object {
        private const val TAG = "InferenceModule"
    }
}

/** Data class describing a single model detection. */
data class Detection(val box: Rect, val label: String, val score: Float)
