import numpy as np
import torch
import onnxruntime as ort
from landmark_detection.search import Similarity_Search


def export_searcher(path, place_ids):
    searcher = Similarity_Search(topk=2)
    boxes = torch.zeros((1, 4), dtype=torch.float32)
    desc = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    db = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0], dtype=torch.float32),
            torch.tensor([0.0, 1.0, 0.0, 0.0, place_ids[1]], dtype=torch.float32),
        ]
    )
    torch.onnx.export(
        searcher,
        (boxes, desc, db),
        path,
        opset_version=16,
        input_names=["boxes", "descriptors", "places_db"],
        output_names=["boxes_out", "scores", "classes"],
        dynamic_axes={
            "boxes": {0: "num_boxes"},
            "descriptors": {0: "num_boxes"},
            "places_db": {0: "db_size"},
            "boxes_out": {0: "num_boxes"},
        "scores": {0: "num_boxes"},
        "classes": {0: "num_boxes"},
        },
        do_constant_folding=False,
    )
    return boxes, desc


def test_dynamic_id(tmp_path):
    onnx_path = tmp_path / "searcher.onnx"
    boxes, desc = export_searcher(onnx_path, [0, 1])
    db_new = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0], dtype=torch.float32),
            torch.tensor([0.0, 1.0, 0.0, 0.0, 3], dtype=torch.float32),
        ]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(
        None,
        {
            "boxes": boxes.numpy(),
            "descriptors": desc.numpy(),
            "places_db": db_new.numpy(),
        },
    )
    assert outs[2].shape[0] == boxes.shape[0]
