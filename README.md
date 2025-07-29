# Landmark Detection Pipeline

This repository contains utilities for exporting and executing a landmark
detection and retrieval pipeline. The exported ONNX model expects **two
inputs**:

1. `image_bgr` – the raw input image in BGR format with shape `(H, W, 3)`
2. `places_db` – database descriptors concatenated with their ``place_id``
   as a tensor with shape ``(N, C + 1)``

## Installation

Install the required packages before running the tests::

    pip install -r requirements.txt

The model internally computes the `[w, h]` tensor from the input image using
dynamic shape operations so it can be traced without constant folding. This
size information is propagated through the detector and extractor so
the post-processing stage can rescale the predictions correctly.

`Pipeline_Landmark_Detection.build_image_database` can be used to generate this
`places_db` tensor. When given a DataFrame mapping each image ``filename`` to
its ``landmark_id`` the function concatenates the identifier to each descriptor
and, optionally, returns the resulting `(N, C + 1)` matrix. If no mapping is
provided, incremental ``image_id`` values will be assigned automatically when
``return_places_db`` is ``True``.

The function `export_places_db` allows storing this array together with a label
mapping on disk::

    from landmark_detection.utils import export_places_db, load_names_from_yaml
    id_to_name = load_names_from_yaml("names.yaml", as_dict=True)
    export_places_db(places_db, id_to_name, "./db")

This will create ``db/places_db.bin`` and ``db/label_map.json``.

## Android AR demo

The sample application in `android_app` now uses a 3‑D marker model for
placing anchors. The file `location.fbx` must be located in
`app/src/main/assets` and is loaded with `ModelLoader` at runtime. When a
detection becomes an anchor the model is instantiated and remains fixed in
world space, replacing the 2‑D icon that was previously drawn on the
overlay.

For build and run instructions, see [`android_app/README.md`](android_app/README.md).
`local.properties` is ignored by Git and must be created locally by Android Studio.
