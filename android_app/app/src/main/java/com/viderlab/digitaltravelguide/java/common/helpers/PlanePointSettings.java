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

/** Manages the plane and point cloud rendering option setting and shared preferences. */
public class PlanePointSettings {
  public static final String SHARED_PREFERENCES_ID = "SHARED_PREFERENCES_PLANE_POINT_OPTIONS";
  public static final String SHARED_PREFERENCES_SHOW_PLANES_AND_POINTS =
      "show_planes_and_points";

  private boolean showPlanesAndPoints = true;
  private SharedPreferences sharedPreferences;

  /** Initializes the current settings based on the saved value. */
  public void onCreate(Context context) {
    sharedPreferences = context.getSharedPreferences(SHARED_PREFERENCES_ID, Context.MODE_PRIVATE);
    showPlanesAndPoints =
        sharedPreferences.getBoolean(SHARED_PREFERENCES_SHOW_PLANES_AND_POINTS, true);
  }

  /** Retrieves whether planes and point clouds should be shown. */
  public boolean getShowPlanesAndPoints() {
    return showPlanesAndPoints;
  }

  public void setShowPlanesAndPoints(boolean enable) {
    if (enable == showPlanesAndPoints) {
      return; // No change.
    }

    // Updates the stored default settings.
    showPlanesAndPoints = enable;
    SharedPreferences.Editor editor = sharedPreferences.edit();
    editor.putBoolean(SHARED_PREFERENCES_SHOW_PLANES_AND_POINTS, showPlanesAndPoints);
    editor.apply();
  }
}
