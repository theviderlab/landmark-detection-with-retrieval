package com.viderlab.digitaltravelguide.kotlin.inference

data class PipelineConfig(
    val name: String,
    val model: String,
    val embeddings: String,
    val labelMap: String? = null,
    val description: String
)
