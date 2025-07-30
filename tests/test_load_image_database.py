import importlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

utils = importlib.import_module("landmark_detection.utils")


def test_load_image_database(tmp_path):
    df = pd.DataFrame({
        "image_name": ["a.jpg", "a.jpg"],
        "bbox": [(0, 0, 1, 1), (0, 0, 1, 1)],
        "class_id": [-1, 0],
        "confidence": [1.0, 0.9],
    })
    descriptors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    df_path = tmp_path / "df.pkl"
    desc_path = tmp_path / "desc.pkl"

    df.to_pickle(df_path)
    with open(desc_path, "wb") as f:
        pickle.dump(descriptors, f)

    df_out, desc_out, places_db = utils.load_image_database(str(df_path), str(desc_path))

    assert df_out.equals(df)
    np.testing.assert_array_equal(desc_out, descriptors)
    expected_ids = np.array([[0], [0]], dtype=np.float32)
    expected_db = np.hstack([descriptors, expected_ids])
    np.testing.assert_array_equal(places_db, expected_db)
