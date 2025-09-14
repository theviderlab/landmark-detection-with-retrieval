/*
 * Copyright 2021 Google LLC
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
package com.viderlab.digitaltravelguide.java.common.helpers;

import android.content.Context;
import android.content.SharedPreferences;

/** Manages the Bounding Box option setting and shared preferences. */
public class BoundingBoxSettings {
  public static final String SHARED_PREFERENCES_ID = "SHARED_PREFERENCES_BBOX_OPTIONS";
  public static final String SHARED_PREFERENCES_SHOW_BOUNDING_BOXES =
      "show_bounding_boxes";
  public static final String SHARED_PREFERENCES_SHOW_DETECTION_LIST =
      "show_detection_list";
  public static final String SHARED_PREFERENCES_SHOW_TEST_DETECTION =
      "show_test_detection";
  private boolean showBoundingBoxes = true;
  private boolean showDetectionList = false;
  private boolean showTestDetection = false;
  private SharedPreferences sharedPreferences;

  /** Initializes the current settings based on the saved value. */
  public void onCreate(Context context) {
    sharedPreferences = context.getSharedPreferences(SHARED_PREFERENCES_ID, Context.MODE_PRIVATE);
    showBoundingBoxes =
        sharedPreferences.getBoolean(SHARED_PREFERENCES_SHOW_BOUNDING_BOXES, true);
    showDetectionList =
        sharedPreferences.getBoolean(SHARED_PREFERENCES_SHOW_DETECTION_LIST, false);
    showTestDetection =
        sharedPreferences.getBoolean(SHARED_PREFERENCES_SHOW_TEST_DETECTION, false);
  }

  /** Retrieves whether bounding boxes should be shown. */
  public boolean getShowBoundingBoxes() {
    return showBoundingBoxes;
  }

  public void setShowBoundingBoxes(boolean enable) {
    if (enable == showBoundingBoxes) {
      return; // No change.
    }

    // Updates the stored default settings.
    showBoundingBoxes = enable;
    SharedPreferences.Editor editor = sharedPreferences.edit();
    editor.putBoolean(SHARED_PREFERENCES_SHOW_BOUNDING_BOXES, showBoundingBoxes);
    editor.apply();
  }

  /** Retrieves whether detection list should be shown. */
  public boolean getShowDetectionList() {
    return showDetectionList;
  }

  public void setShowDetectionList(boolean enable) {
    if (enable == showDetectionList) {
      return; // No change.
    }

    // Updates the stored default settings.
    showDetectionList = enable;
    SharedPreferences.Editor editor = sharedPreferences.edit();
    editor.putBoolean(SHARED_PREFERENCES_SHOW_DETECTION_LIST, showDetectionList);
    editor.apply();
  }

  /** Retrieves whether test detection should be shown. */
  public boolean getShowTestDetection() {
    return showTestDetection;
  }

  public void setShowTestDetection(boolean enable) {
    if (enable == showTestDetection) {
      return; // No change.
    }

    // Updates the stored default settings.
    showTestDetection = enable;
    SharedPreferences.Editor editor = sharedPreferences.edit();
    editor.putBoolean(SHARED_PREFERENCES_SHOW_TEST_DETECTION, showTestDetection);
    editor.apply();
  }
}
