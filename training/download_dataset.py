"""
Download the traffic detection dataset from Roboflow.

Usage:
    pip install roboflow
    python training/download_dataset.py

This downloads ~1500 labeled traffic images (car, truck, bus)
in YOLO format, ready for training. The dataset is split into
train/valid/test automatically by Roboflow.

You need a free Roboflow API key:
1. Go to https://app.roboflow.com → Settings → API Key
2. Copy it
3. Either pass it as argument or set ROBOFLOW_API_KEY env var
"""

import os
import sys

try:
    from roboflow import Roboflow
except ImportError:
    print("roboflow package not found. Install it with:")
    print("  pip install roboflow")
    sys.exit(1)


def download_dataset(api_key: str = None, output_dir: str = "data"):
    """Download the traffic dataset from Roboflow."""

    # Get API key from argument, env var, or prompt
    if api_key is None:
        api_key = os.getenv("ROBOFLOW_API_KEY")
    if api_key is None:
        api_key = input("Enter your Roboflow API key: ").strip()

    if not api_key:
        print("Error: No API key provided.")
        print("Get one free at https://app.roboflow.com → Settings → API Key")
        sys.exit(1)

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    print("Downloading traffic dataset...")
    project = rf.workspace("yolo-iuil0").project("traffic-4zbse")
    version = project.version(9)

    # Download in YOLOv8 format — this is what ultralytics expects
    dataset = version.download("yolov8", location=f"{output_dir}/traffic")

    print(f"\nDataset downloaded to: {output_dir}/traffic/")
    print(f"  Train images: {output_dir}/traffic/train/images/")
    print(f"  Valid images: {output_dir}/traffic/valid/images/")
    print(f"  Test images:  {output_dir}/traffic/test/images/")
    print(f"  Config file:  {output_dir}/traffic/data.yaml")
    print("\nYou're ready to train! Run:")
    print("  python training/train.py")

    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download traffic dataset from Roboflow")
    parser.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--output-dir", default="data", help="Directory to save dataset")
    args = parser.parse_args()

    download_dataset(api_key=args.api_key, output_dir=args.output_dir)
