import importlib
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _setup_dummy_modules():
    dummy = types.ModuleType("dummy")
    for name in ["cv2", "ultralytics", "onnxruntime"]:
        sys.modules[name] = dummy

    class FakeTensor(np.ndarray):
        def numpy(self):
            return np.asarray(self)

    def _as_tensor(x):
        arr = np.asarray(x).view(FakeTensor)
        return arr

    def _matmul(a, b):
        res = np.matmul(np.asarray(a), np.asarray(b)).view(FakeTensor)
        return res

    torch_mod = types.ModuleType("torch")
    torch_mod.as_tensor = _as_tensor
    torch_mod.matmul = _matmul
    sys.modules["torch"] = torch_mod

    onnx_mod = types.ModuleType("onnx")
    compose_mod = types.ModuleType("onnx.compose")
    helper_mod = types.ModuleType("onnx.helper")
    onnx_mod.compose = compose_mod
    onnx_mod.helper = helper_mod
    sys.modules["onnx"] = onnx_mod
    sys.modules["onnx.compose"] = compose_mod
    sys.modules["onnx.helper"] = helper_mod

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = object
    sys.modules["ultralytics"] = ultra

    pre = types.ModuleType("landmark_detection.preprocess")
    pre.PreprocessModule = object
    sys.modules["landmark_detection.preprocess"] = pre

    ext = types.ModuleType("landmark_detection.extract")
    ext.CVNet_SG = object
    sys.modules["landmark_detection.extract"] = ext

    post = types.ModuleType("landmark_detection.postprocess")
    post.PostprocessModule = object
    sys.modules["landmark_detection.postprocess"] = post

    search = types.ModuleType("landmark_detection.search")
    search.Similarity_Search = object
    sys.modules["landmark_detection.search"] = search


def test_build_image_database_keep_full(tmp_path):
    _setup_dummy_modules()
    pl = importlib.import_module("landmark_detection.pipeline")

    class DummyPipeline(pl.Pipeline_Landmark_Detection):
        def __init__(self, boxes, descriptors):
            self._boxes = np.asarray(boxes, dtype=np.float32)
            self._descs = np.asarray(descriptors, dtype=np.float32)

        def preprocess(self, image_path, places_db):
            return None, (1, 1), places_db

        def detect(self, img_proc, places_db, orig_size):
            return self._boxes, img_proc, places_db, orig_size

        def extract(self, detections, img_proc, places_db, orig_size):
            return self._boxes, self._descs

    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
        ],
        dtype=np.float32,
    )
    descriptors = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_text("dummy")

    df_path = tmp_path / "df.pkl"
    desc_path = tmp_path / "desc.pkl"

    pipe = DummyPipeline(boxes, descriptors)
    df, _ = pipe.build_image_database(
        image_folder=str(img_dir),
        df_pickle_path=str(df_path),
        descriptor_pickle_path=str(desc_path),
        min_sim=0.9,
        keep_full_img=False,
    )
    assert -1 not in df["class_id"].values

    df_path2 = tmp_path / "df2.pkl"
    desc_path2 = tmp_path / "desc2.pkl"

    pipe2 = DummyPipeline(boxes, descriptors)
    df2, _ = pipe2.build_image_database(
        image_folder=str(img_dir),
        df_pickle_path=str(df_path2),
        descriptor_pickle_path=str(desc_path2),
        min_sim=0.9,
        keep_full_img=True,
    )
    assert -1 in df2["class_id"].values
