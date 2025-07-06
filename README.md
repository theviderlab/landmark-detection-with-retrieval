# Landmark Detection Pipeline

This repository contains utilities for exporting and executing a landmark
detection and retrieval pipeline. The exported ONNX model expects **two
inputs**:

1. `image_bgr` – the raw input image in BGR format with shape `(H, W, 3)`
2. `places_db` – database descriptors concatenated with their ``place_id``
   as a tensor with shape ``(N, C + 1)``

The model internally computes the `[w, h]` tensor from the input image using
dynamic shape operations so it can be traced without constant folding. This
size information is propagated through the detector and extractor so
the post-processing stage can rescale the predictions correctly.
