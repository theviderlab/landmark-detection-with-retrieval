/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.viderlab.digitaltravelguide.kotlin.helloar

import android.content.res.Resources
import android.opengl.GLSurfaceView
import android.view.View
import android.widget.RelativeLayout
import android.widget.ImageButton
import android.widget.PopupMenu
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.TextView
import android.widget.AdapterView
import androidx.appcompat.app.AlertDialog
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Config
import com.viderlab.digitaltravelguide.java.common.helpers.SnackbarHelper

/** Contains UI elements for Hello AR. */
class HelloArView(val activity: HelloArActivity) : DefaultLifecycleObserver {
  private var inferenceEnabled = true
  val root = View.inflate(activity, R.layout.activity_main, null) as RelativeLayout
  val surfaceView = root.findViewById<GLSurfaceView>(R.id.surfaceview)
    private val boundingBoxOverlay = BoundingBoxOverlay(activity)
    private val markerOverlay = MarkerOverlay(activity)
    private val detectionListOverlay = DetectionListOverlay(activity)
  init {
    root.addView(
      boundingBoxOverlay,
      RelativeLayout.LayoutParams(
        RelativeLayout.LayoutParams.MATCH_PARENT,
        RelativeLayout.LayoutParams.MATCH_PARENT
      )
    )
    root.addView(
      markerOverlay,
      RelativeLayout.LayoutParams(
        RelativeLayout.LayoutParams.MATCH_PARENT,
        RelativeLayout.LayoutParams.MATCH_PARENT
      )
    )
    root.addView(
      detectionListOverlay,
      RelativeLayout.LayoutParams(
        RelativeLayout.LayoutParams.MATCH_PARENT,
        RelativeLayout.LayoutParams.MATCH_PARENT
      )
    )
  }
  val settingsButton =
    root.findViewById<ImageButton>(R.id.settings_button).apply {
      setOnClickListener { v ->
        PopupMenu(activity, v).apply {
          setOnMenuItemClickListener { item ->
            when (item.itemId) {
              R.id.depth_settings -> {
                launchDepthSettingsMenuDialog()
                true
              }
              R.id.instant_placement_settings -> {
                launchInstantPlacementSettingsMenuDialog()
                true
              }
              R.id.bbox_settings -> {
                launchBoundingBoxSettingsMenuDialog()
                true
              }
              R.id.pipeline_settings -> {
                launchPipelineSettingsMenuDialog()
                true
              }
              R.id.plane_point_settings -> {
                launchPlanePointSettingsMenuDialog()
                true
              }
              R.id.clear_markers -> {
                activity.renderer.clearMarkers()
                true
              }
              else -> false
            }
          }
          inflate(R.menu.settings_menu)
          show()
        }
      }
    }

  val inferenceButton =
    root.findViewById<ImageButton>(R.id.inference_toggle_button).apply {
      setOnClickListener {
        inferenceEnabled = !inferenceEnabled
        activity.renderer.setInferenceEnabled(inferenceEnabled)
        setImageResource(
          if (inferenceEnabled) R.drawable.ic_inference_on
          else R.drawable.ic_inference_off
        )
      }
    }

  val session
    get() = activity.arCoreSessionHelper.session

  val snackbarHelper = SnackbarHelper()

  fun showDebugBoundingBoxes(bboxes: List<LabeledBoundingBox>) {
    if (!activity.boundingBoxSettings.showBoundingBoxes) {
      boundingBoxOverlay.clearBoundingBoxes()
      return
    }
    if (bboxes.isEmpty()) {
      boundingBoxOverlay.clearBoundingBoxes()
    } else {
      boundingBoxOverlay.showBoundingBoxes(bboxes)
    }
  }

  fun showDetectionList(detections: List<LabeledBoundingBox>) {
    if (!activity.boundingBoxSettings.showDetectionList) {
      detectionListOverlay.clearDetections()
      return
    }
    if (detections.isEmpty()) {
      detectionListOverlay.clearDetections()
    } else {
      detectionListOverlay.showDetections(detections)
    }
  }

  fun clearDetectionList() {
    detectionListOverlay.clearDetections()
  }


  fun updateAnchors(markers: List<MarkerInfo>, drawBoxes: Boolean = false) {
    val overlayMarkers = markers.map { MarkerOverlay.Marker(it.x, it.y, it.label) }
    markerOverlay.updateAnchors(overlayMarkers, drawBoxes)
    val draw = drawBoxes && activity.boundingBoxSettings.showBoundingBoxes
    if (draw) {
      val boxes =
        markers.mapNotNull { info -> info.bbox?.let { bbox -> LabeledBoundingBox(bbox, info.label, 1f) } }
      if (boxes.isEmpty()) {
        boundingBoxOverlay.clearBoundingBoxes()
      } else {
        boundingBoxOverlay.showBoundingBoxes(boxes)
      }
    }
  }

  override fun onResume(owner: LifecycleOwner) {
    surfaceView.onResume()
  }

  override fun onPause(owner: LifecycleOwner) {
    surfaceView.onPause()
  }

  /**
   * Shows a pop-up dialog when the first marker is placed automatically in HelloARRenderer,
   * determining whether the user wants to enable depth-based occlusion. The result of this dialog
   * can be retrieved with DepthSettings.useDepthForOcclusion().
   */
  fun showOcclusionDialogIfNeeded() {
    val session = session ?: return
    val isDepthSupported = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
    if (!activity.depthSettings.shouldShowDepthEnableDialog() || !isDepthSupported) {
      return // Don't need to show dialog.
    }

    // Asks the user whether they want to use depth-based occlusion.
    AlertDialog.Builder(activity)
      .setTitle(R.string.options_title_with_depth)
      .setMessage(R.string.depth_use_explanation)
      .setPositiveButton(R.string.button_text_enable_depth) { _, _ ->
        activity.depthSettings.setUseDepthForOcclusion(true)
      }
      .setNegativeButton(R.string.button_text_disable_depth) { _, _ ->
        activity.depthSettings.setUseDepthForOcclusion(false)
      }
      .show()
  }

  private fun launchPipelineSettingsMenuDialog() {
    val repository = activity.pipelineRepository
    val configs = repository.pipelines
    val names = configs.map { it.name }
    val selectedName = activity.pipelineSettings.getSelectedPipeline()

    val dialogView = View.inflate(activity, R.layout.pipeline_settings_dialog, null)
    val spinner = dialogView.findViewById<Spinner>(R.id.pipeline_spinner)
    val descriptionView = dialogView.findViewById<TextView>(R.id.pipeline_description)

    val adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_dropdown_item, names)
    spinner.adapter = adapter

    var index = names.indexOf(selectedName).takeIf { it >= 0 } ?: 0
    if (configs.isNotEmpty()) {
      spinner.setSelection(index)

      fun updateDescription(position: Int) {
        val config = configs[position]
        descriptionView.text = config.description
      }

      updateDescription(index)

      spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
          index = position
          updateDescription(position)
        }

        override fun onNothingSelected(parent: AdapterView<*>) {}
      }
    } else {
      spinner.isEnabled = false
    }

    val dialog =
        AlertDialog.Builder(activity)
            .setTitle(R.string.options_title_pipeline)
            .setView(dialogView)
            .setPositiveButton(R.string.done) { _, _ ->
              if (configs.isNotEmpty()) {
                val config = configs[index]
                activity.pipelineSettings.setSelectedPipeline(config.name)
                activity.renderer.setPipeline(config)
              }
            }
            .create()
    dialog.show()
    dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = configs.isNotEmpty()
  }

  private fun launchInstantPlacementSettingsMenuDialog() {
    val resources = activity.resources
    val strings = resources.getStringArray(R.array.instant_placement_options_array)
    val checked = booleanArrayOf(activity.instantPlacementSettings.isInstantPlacementEnabled)
    AlertDialog.Builder(activity)
      .setTitle(R.string.options_title_instant_placement)
      .setMultiChoiceItems(strings, checked) { _, which, isChecked -> checked[which] = isChecked }
      .setPositiveButton(R.string.done) { _, _ ->
        val session = session ?: return@setPositiveButton
        activity.instantPlacementSettings.isInstantPlacementEnabled = checked[0]
        activity.configureSession(session)
      }
      .show()
  }

  private fun launchBoundingBoxSettingsMenuDialog() {
    val resources = activity.resources
    val strings = resources.getStringArray(R.array.bbox_options_array)
    val checked = booleanArrayOf(
      activity.boundingBoxSettings.showBoundingBoxes,
      activity.boundingBoxSettings.showDetectionList,
      activity.boundingBoxSettings.showTestDetection
    )
    AlertDialog.Builder(activity)
      .setTitle(R.string.options_title_bbox)
      .setMultiChoiceItems(strings, checked) { _, which, isChecked -> checked[which] = isChecked }
      .setPositiveButton(R.string.done) { _, _ ->
        activity.boundingBoxSettings.showBoundingBoxes = checked[0]
        activity.boundingBoxSettings.showDetectionList = checked[1]
        activity.boundingBoxSettings.showTestDetection = checked[2]
        if (!checked[0]) {
          boundingBoxOverlay.clearBoundingBoxes()
        }
        if (!checked[1]) {
          detectionListOverlay.clearDetections()
        }
        if (!checked[2]) {
          boundingBoxOverlay.clearBoundingBoxes()
          detectionListOverlay.clearDetections()
        }
      }
      .show()
  }

  private fun launchPlanePointSettingsMenuDialog() {
    val resources = activity.resources
    val strings = resources.getStringArray(R.array.plane_point_options_array)
    val checked = booleanArrayOf(activity.planePointSettings.showPlanesAndPoints)
    AlertDialog.Builder(activity)
      .setTitle(R.string.options_title_plane_point)
      .setMultiChoiceItems(strings, checked) { _, which, isChecked -> checked[which] = isChecked }
      .setPositiveButton(R.string.done) { _, _ ->
        activity.planePointSettings.showPlanesAndPoints = checked[0]
      }
      .show()
  }

  /** Shows checkboxes to the user to facilitate toggling of depth-based effects. */
  private fun launchDepthSettingsMenuDialog() {
    val session = session ?: return

    // Shows the dialog to the user.
    val resources: Resources = activity.resources
    val checkboxes =
      booleanArrayOf(
        activity.depthSettings.useDepthForOcclusion(),
        activity.depthSettings.depthColorVisualizationEnabled()
      )
    if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
      // With depth support, the user can select visualization options.
      val stringArray = resources.getStringArray(R.array.depth_options_array)
      AlertDialog.Builder(activity)
        .setTitle(R.string.options_title_with_depth)
        .setMultiChoiceItems(stringArray, checkboxes) { _, which, isChecked ->
          checkboxes[which] = isChecked
        }
        .setPositiveButton(R.string.done) { _, _ ->
          activity.depthSettings.setUseDepthForOcclusion(checkboxes[0])
          activity.depthSettings.setDepthColorVisualizationEnabled(checkboxes[1])
        }
        .show()
    } else {
      // Without depth support, no settings are available.
      AlertDialog.Builder(activity)
        .setTitle(R.string.options_title_without_depth)
        .setPositiveButton(R.string.done) { _, _ -> /* No settings to apply. */ }
        .show()
    }
  }
}
