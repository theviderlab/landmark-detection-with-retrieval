import yaml
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
from landmark_detection.utils import load_names_from_yaml


def test_load_names_list(tmp_path):
    path = tmp_path / "names.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"names": {0: "name0", 1: "name1"}}, f)

    names = load_names_from_yaml(path)
    assert names == ["name0", "name1"]


def test_load_names_dict(tmp_path):
    path = tmp_path / "names.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"names": {0: "name0", 1: "name1"}}, f)

    names_dict = load_names_from_yaml(path, as_dict=True)
    assert names_dict == {0: "name0", 1: "name1"}

