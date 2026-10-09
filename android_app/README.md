# Android AR demo

An Android app that recognises the landmarks of Cáceres from the camera and anchors a marker on each one in augmented reality. It is based on the ARCore `hello_ar_kotlin` sample, which provides the rendering, session handling and helpers under `java/.../common` (Apache 2.0, Google LLC).

## How it works

- `HelloArRenderer` takes a camera frame, converts it from YUV to RGB and passes it to `InferenceModule` on a background thread.
- `InferenceModule` runs the end-to-end ONNX pipeline with [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-java.html). The inputs are the BGR image and the embedding database `places_db.bin`. The outputs are the boxes, scores and landmark ids, already in image coordinates.
- `label_map.json` maps each landmark id to its name.
- Each detection is hit-tested against the AR scene and turned into an ARCore anchor, so the marker stays fixed in world space.
- `pipelines.yaml` lists the available models. You can add entries to compare models exported with different parameters.

## Assets

`app/src/main/assets/pipeline/` contains:

| File | In the repo | Description |
|---|---|---|
| `places_db.bin` | ✅ | Embeddings of the Cáceres database (945 × (2048 + 1) float32 values, after an `int32` header with the number of rows and columns) |
| `label_map.json` | ✅ | Landmark id → name |
| `pipeline.onnx` | ❌ (~105 MB) | Must be generated, see below |

## Build and run

1. Generate the model from the repository root. This requires the pretrained weights described in the [main README](../README.md#2-download-the-pretrained-weights):

   ```bash
   python scripts/export_onnx.py --config configs/app.json \
       --output android_app/app/src/main/assets/pipeline/pipeline.onnx
   ```

2. Open `android_app/` in Android Studio. Android Studio creates `local.properties` with your SDK path.
3. Run the app on an [ARCore-compatible device](https://developers.google.com/ar/devices) with Android 8.0 (API 26) or later.

Requirements: JDK 17, Android SDK 36, ARCore 1.50, ONNX Runtime Android 1.22.
