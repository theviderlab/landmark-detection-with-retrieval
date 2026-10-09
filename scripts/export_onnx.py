"""Export the end-to-end ONNX pipeline from a JSON configuration.

Usage (from the repository root):

    python scripts/export_onnx.py --config configs/app.json \
        --output android_app/app/src/main/assets/pipeline/pipeline.onnx

The intermediate models (preprocess, detector, extractor, searcher and
postprocess) are written to ``landmark_detection/models/``.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from landmark_detection.pipeline import Pipeline_Landmark_Detection


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/app.json", help="JSON file with the pipeline parameters")
    parser.add_argument("--output", default=None, help="Optional path where the fused pipeline ONNX is copied")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["image_dim"] = tuple(config["image_dim"])

    pipeline = Pipeline_Landmark_Detection(**config)
    print(f"Pipeline exported to {pipeline.pipeline_onnx_path}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        shutil.copyfile(pipeline.pipeline_onnx_path, args.output)
        print(f"Copied to {args.output}")


if __name__ == "__main__":
    main()
