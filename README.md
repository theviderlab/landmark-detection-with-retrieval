# Landmark Detection Pipeline

This repository contains utilities for exporting and executing a landmark
detection and retrieval pipeline. The exported ONNX model expects **two
inputs**:

1. `image_bgr` – the raw input image in BGR format with shape `(H, W, 3)`
2. `orig_size` – a tensor `[w, h]` with the original width and height of
   the image.

The `orig_size` tensor allows dynamic resolutions without relying on ONNX
`Shape` operations.
