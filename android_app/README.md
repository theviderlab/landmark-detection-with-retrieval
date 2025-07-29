# Android AR Landmark Demo

This module contains a minimal Kotlin app showing how to run the landmark detection pipeline on device using ARCore. The code is located under `app/src/main/java` and reuses the same `YoloDetector` implementation from the `android_app` project.

## Overview

The app loads `pipeline.onnx` together with the descriptor database (`places_db.bin`) and the category mapping stored in `label_map.json`. At runtime each camera frame is processed with ONNX Runtime to obtain landmark detections. The label map is used to translate class indices to human‑readable names.

An `ARSceneView` is configured with plane finding disabled and automatic depth mode:

```kotlin
sceneView.session?.let { session ->
    val config = Config(session).apply {
        planeFindingMode = Config.PlaneFindingMode.DISABLED
        depthMode = Config.DepthMode.AUTOMATIC
    }
    session.configure(config)
}
```

Detections are shown on top of the camera feed using `BoxOverlay`. When a detection is first seen it becomes an ARCore anchor so that a `location.glb` marker stays fixed in world space. Anchors are updated or removed if new detections move too much or disappear.

The minimum supported Android version is 24 and a device must support ARCore with the Depth API.

## Building and running

1. Open `android_app` in Android Studio.
2. Copy `pipeline.onnx`, `places_db.bin`, `label_map.json` and `location.glb` from `android_app/app/src/main/assets` into the `assets` folder of this module.
3. Connect an ARCore‑capable Android device (API 24+).
4. Build and run the `app` configuration from Android Studio or execute `./gradlew installDebug`.

