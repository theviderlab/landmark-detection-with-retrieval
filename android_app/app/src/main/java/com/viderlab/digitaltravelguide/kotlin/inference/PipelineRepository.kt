package com.viderlab.digitaltravelguide.kotlin.inference

import android.content.Context
import android.util.Log
import kotlin.text.Charsets
import org.yaml.snakeyaml.Yaml

/**
 * Loads pipeline configurations from the `pipelines.yaml` asset and
 * validates that referenced asset files exist.
 */
class PipelineRepository(context: Context) {
    private val assetManager = context.assets

    val pipelines: List<PipelineConfig>

    init {
        pipelines = loadPipelines()
    }

    private fun loadPipelines(): List<PipelineConfig> {
        val yaml = Yaml()
        return try {
            val root: Map<String, Any?>? =
                assetManager.open("pipelines.yaml").use { input -> yaml.load(input.reader(Charsets.UTF_8)) }
            val list = root?.get("pipelines") as? List<*> ?: run {
                Log.e(TAG, "pipelines.yaml is empty. Using dummy pipeline")
                return listOf(
                    PipelineConfig(
                        name = "dummy",
                        model = "",
                        embeddings = "",
                        description = "Default dummy pipeline"
                    )
                )
            }

            val configs = mutableListOf<PipelineConfig>()
            for (item in list) {
                val map = item as? Map<*, *> ?: continue
                val name = map["name"] as? String ?: continue
                val model = map["model"] as? String ?: continue
                val embeddings = map["embeddings"] as? String ?: continue
                val labelMap = map["label_map"] as? String
                val description = map["description"] as? String ?: ""

                if (!exists(model) || !exists(embeddings) || (labelMap != null && !exists(labelMap))) {
                    Log.w(TAG, "Skipping pipeline '$name' due to missing assets")
                    continue
                }

                configs += PipelineConfig(
                    name = name,
                    model = model,
                    embeddings = embeddings,
                    labelMap = labelMap,
                    description = description
                )
            }
            if (configs.isEmpty()) {
                Log.e(TAG, "No valid pipelines found")
            }
            configs
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse pipelines.yaml", e)
            listOf(
                PipelineConfig(
                    name = "dummy",
                    model = "",
                    embeddings = "",
                    description = "Default dummy pipeline"
                )
            )
        }
    }

    private fun exists(path: String): Boolean = try {
        assetManager.open(path).close()
        true
    } catch (e: Exception) {
        false
    }

    fun get(name: String): PipelineConfig? = pipelines.firstOrNull { it.name == name }

    companion object {
        private const val TAG = "PipelineRepository"
    }
}
