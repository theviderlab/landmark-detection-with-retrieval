# Landmark Detection Pipeline

This repository contains utilities for exporting and executing a landmark
detection and retrieval pipeline. The exported ONNX model expects **one
input**:

1. `image_bgr` – the raw input image in BGR format with shape `(H, W, 3)`

The model internally computes the `[w, h]` tensor that represents the
original image size and propagates it through the detector and extractor so
the post-processing stage can rescale the predictions correctly.
