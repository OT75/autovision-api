"""
Download the traffic detection dataset from Roboflow.

Usage:
    python training/download_dataset.py --api-key YOUR_KEY
    # or
    ROBOFLOW_API_KEY=your_key python training/download_dataset.py

Get your free API key at https://app.roboflow.com → Settings → API Key
"""

import argparse
import os
import sys


def download_dataset(api_key: str, output_dir: str = "data"):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow first:  pip install roboflow")
        sys.exit(1)

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace("yolo-iuil0").project("traffic-4zbse").version(9).download(
        "yolov8", location=f"{output_dir}/traffic"
    )

    print(f"\nDataset saved to: {output_dir}/traffic/")
    print("Next step:  python training/train.py --config training/configs/exp1_baseline.yaml")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set ROBOFLOW_API_KEY env var")
        sys.exit(1)

    download_dataset(api_key=args.api_key, output_dir=args.output_dir)
