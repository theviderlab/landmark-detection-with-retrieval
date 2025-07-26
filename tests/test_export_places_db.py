import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# avoid heavy deps
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

from landmark_detection.utils import export_places_db

import numpy as np

def test_export_places_db(tmp_path):
    db = np.array([
        [0.1, 0.2, 0],
        [0.3, 0.4, 1],
    ], dtype=np.float32)
    label_map = {0: "a", 1: "b"}
    export_places_db(db, label_map, tmp_path)

    bin_path = tmp_path / "places_db.bin"
    assert bin_path.is_file()
    with open(bin_path, "rb") as f:
        data = f.read()
    n, c = db.shape
    expected_size = 8 + 4 * n * c
    assert len(data) == expected_size

    label_path = tmp_path / "label_map.json"
    with open(label_path) as f:
        loaded = json.load(f)
    expected_label_map = {str(k): v for k, v in label_map.items()}
    assert loaded == expected_label_map

