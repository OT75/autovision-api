"""
Train YOLO on the traffic dataset with MLflow experiment tracking.

Usage:
    python training/train.py --config training/configs/exp1_baseline.yaml
    python training/train.py --config training/configs/exp2_high_res.yaml
    python training/train.py --config training/configs/exp3_heavy_aug.yaml
"""

import argparse
import csv
import os
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "autovision-traffic-detection"

TRACKED_PARAMS = ["base_model", "epochs", "imgsz", "batch", "lr0", "lrf", "mosaic", "mixup"]

TRACKED_METRICS = {
    "metrics_mAP50B":     "mAP50",
    "metrics_mAP50-95B":  "mAP50-95",
    "metrics_precisionB": "precision",
    "metrics_recallB":    "recall",
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_final_metrics(results_dir: Path) -> dict[str, float]:
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    metrics = {}
    for raw_key, value in rows[-1].items():
        clean_key = raw_key.strip().replace("/", "_").replace("(", "").replace(")", "")
        try:
            metrics[clean_key] = float(value.strip())
        except (ValueError, AttributeError):
            pass
    return metrics


def get_or_create_experiment(client, name: str, artifact_uri: str) -> str:
    """
    Get the experiment ID by name.
    - If it doesn't exist: create it fresh.
    - If it exists but is soft-deleted: restore it first, then return it.
    - If it exists and is active: return it as-is.
    """
    # Search ALL experiments including deleted ones (view_type=3)
    experiments = client.search_experiments(view_type=3)
    for exp in experiments:
        if exp.name == name:
            if exp.lifecycle_stage == "deleted":
                print(f"  Experiment '{name}' was soft-deleted — restoring it...")
                client.restore_experiment(exp.experiment_id)
            return exp.experiment_id

    # Doesn't exist at all — create fresh with correct artifact location
    print(f"  Creating experiment '{name}'...")
    return client.create_experiment(name, artifact_location=artifact_uri)


def train(config_path: str):
    cfg = load_config(config_path)

    # Disable Ultralytics' built-in MLflow callback before YOLO loads.
    # It tries to create its own 'runs/train' experiment and conflicts with ours.
    os.environ["MLFLOW_TRACKING_URI"] = ""

    # Set our tracking URI directly on the mlflow client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    artifact_uri = MLFLOW_TRACKING_URI.replace("http://", "mlflow-artifacts:/").replace("https://", "mlflow-artifacts:/")
    client = mlflow.tracking.MlflowClient()

    exp_id = get_or_create_experiment(client, EXPERIMENT_NAME, artifact_uri)
    mlflow.set_experiment(experiment_id=exp_id)

    print(f"Starting run: {cfg.get('run_name', 'unnamed')}")
    print(f"  Model: {cfg['base_model']}  |  Epochs: {cfg['epochs']}  |  imgsz: {cfg['imgsz']}")
    print(f"  MLflow: {MLFLOW_TRACKING_URI}")

    with mlflow.start_run(run_name=cfg.get("run_name")) as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id[:8]}")

        mlflow.log_params({k: cfg[k] for k in TRACKED_PARAMS if k in cfg})

        model = YOLO(cfg["base_model"])
        results = model.train(
            data=cfg["data_yaml"],
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            lr0=cfg["lr0"],
            lrf=cfg["lrf"],
            mosaic=cfg.get("mosaic", 1.0),
            mixup=cfg.get("mixup", 0.0),
            hsv_h=cfg.get("hsv_h", 0.015),
            hsv_s=cfg.get("hsv_s", 0.7),
            hsv_v=cfg.get("hsv_v", 0.4),
            fliplr=cfg.get("fliplr", 0.5),
            project="runs/train",
            name=run_id,
            exist_ok=True,
            verbose=True,
        )

        results_dir = (
            Path(results.save_dir) if hasattr(results, "save_dir")
            else Path(f"runs/train/{run_id}")
        )
        print(f"  YOLO output dir: {results_dir}")

        # Log 4 key final metrics
        all_metrics = read_final_metrics(results_dir)
        final_metrics = {
            display: all_metrics[yolo_key]
            for yolo_key, display in TRACKED_METRICS.items()
            if yolo_key in all_metrics
        }

        if final_metrics:
            mlflow.log_metrics(final_metrics)
            print("\nFinal metrics:")
            for name, value in final_metrics.items():
                print(f"  {name}: {value:.4f}")
        else:
            print("\nWarning: could not read metrics from results.csv")
            print(f"  Expected at: {results_dir / 'results.csv'}")

        # Upload best.pt
        best_pt = results_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"\nUploading model ({best_pt.stat().st_size / 1e6:.1f} MB)...")
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
            print("  ✓ weights/best.pt uploaded")
        else:
            print(f"\nWarning: best.pt not found at {best_pt}")

        # Upload training plots
        plots_uploaded = 0
        for plot in results_dir.glob("*.png"):
            mlflow.log_artifact(str(plot), artifact_path="plots")
            plots_uploaded += 1
        if plots_uploaded:
            print(f"  ✓ {plots_uploaded} plots uploaded")

        print(f"\nDone. View at {MLFLOW_TRACKING_URI}  (run: {run_id[:8]})")
        return run_id, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    train(args.config)
