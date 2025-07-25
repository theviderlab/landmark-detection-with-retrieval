import json
import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_save_evaluation_result(tmp_path):
    # create dummy modules so benchmark.evaluation can be imported without
    # heavy optional dependencies
    dummy = types.ModuleType("dummy")
    sys.modules.setdefault("cv2", dummy)
    mpl = types.ModuleType("matplotlib")
    mpl_pyplot = types.ModuleType("matplotlib.pyplot")
    mpl.patches = types.ModuleType("matplotlib.patches")
    mpl.patches.Rectangle = object
    mpl.pyplot = mpl_pyplot
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", mpl_pyplot)
    sys.modules.setdefault("matplotlib.patches", mpl.patches)
    sys.modules.setdefault("pandas", dummy)
    sys.modules.setdefault("torch", dummy)
    sys.modules.setdefault("torch.nn", dummy)
    mod = types.ModuleType("search")
    mod.Similarity_Search = object
    sys.modules.setdefault("landmark_detection.search", mod)

    evaluation = importlib.import_module("benchmark.evaluation")

    path = tmp_path / "results" / "eval.json"

    res1 = {"a": 1}
    cfg1 = {"b": 2}
    evaluation.save_evaluation_result(res1, str(path), cfg1)

    with open(path) as f:
        data = json.load(f)
    assert data == [{"results": res1, "config": cfg1}]

    res2 = {"a": 3}
    cfg2 = {"b": 4}
    evaluation.save_evaluation_result(res2, str(path), cfg2)

    with open(path) as f:
        data = json.load(f)
    assert data == [
        {"results": res1, "config": cfg1},
        {"results": res2, "config": cfg2},
    ]

