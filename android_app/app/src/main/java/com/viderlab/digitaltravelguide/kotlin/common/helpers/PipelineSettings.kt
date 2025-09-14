package com.viderlab.digitaltravelguide.kotlin.common.helpers

import android.content.Context
import android.content.SharedPreferences
import com.viderlab.digitaltravelguide.kotlin.inference.PipelineRepository

class PipelineSettings {
  companion object {
    const val SHARED_PREFERENCES_PIPELINE = "SHARED_PREFERENCES_PIPELINE"
    const val SHARED_PREFERENCES_SELECTED_PIPELINE = "selected_pipeline"
  }

  private lateinit var sharedPreferences: SharedPreferences
  private lateinit var selectedPipeline: String

  fun onCreate(context: Context) {
    sharedPreferences =
        context.getSharedPreferences(SHARED_PREFERENCES_PIPELINE, Context.MODE_PRIVATE)
    val repo = PipelineRepository(context)
    val default = repo.pipelines.firstOrNull()?.name ?: ""
    var name =
        sharedPreferences.getString(SHARED_PREFERENCES_SELECTED_PIPELINE, default) ?: default
    if (repo.get(name) == null) {
      name = repo.pipelines.firstOrNull()?.name ?: ""
    }
    selectedPipeline = name
    sharedPreferences.edit().putString(SHARED_PREFERENCES_SELECTED_PIPELINE, selectedPipeline).apply()
  }

  fun getSelectedPipeline(): String = selectedPipeline

  fun setSelectedPipeline(name: String) {
    if (name == selectedPipeline) {
      return
    }
    selectedPipeline = name
    sharedPreferences.edit().putString(SHARED_PREFERENCES_SELECTED_PIPELINE, selectedPipeline).apply()
  }
}

