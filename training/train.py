"""
Train YOLO on the traffic dataset with MLflow experiment tracking.

Usage:
    python training/train.py                           # default config
    python training/train.py --epochs 30 --imgsz 416   # custom config
    python training/train.py --config training/configs/experiment_2.yaml

This script:
1. Starts an MLflow run
2. Logs all training parameters (lr, epochs, imgsz, batch, augmentation)
3. Trains YOLO on the traffic dataset
4. Logs metrics (mAP50, mAP50-95, precision, recall) to MLflow
5. Logs the trained model as an MLflow artifact
6. Optionally registers the model in MLflow Model Registry

After training, open MLflow UI at http://localhost:5000 to compare runs.

Why MLflow?
    Without it, you train 5 models and forget which learning rate
    gave the best results. MLflow tracks everything automatically
    so you can compare runs side-by-side and pick the winner.
"""

import argparse
import os
import shutil
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO


# --- MLflow setup ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "autovision-traffic-detection"


def train(
    # Model
    base_model: str = "yolov8n.pt",
    # Data
    data_yaml: str = "data/traffic/data.yaml",
    # Training hyperparameters
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.01,
    lrf: float = 0.01,
    # Augmentation
    augment: bool = True,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    # MLflow
    run_name: str = None,
    register_model: bool = False,
    # Config file (overrides all above if provided)
    config: str = None,
):
    """Train YOLO with full MLflow tracking."""

    # --- Load config file if provided ---
    if config and os.path.exists(config):
        print(f"Loading config from {config}")
        with open(config) as f:
            cfg = yaml.safe_load(f)
        # Override defaults with config values
        base_model = cfg.get("base_model", base_model)
        data_yaml = cfg.get("data_yaml", data_yaml)
        epochs = cfg.get("epochs", epochs)
        imgsz = cfg.get("imgsz", imgsz)
        batch = cfg.get("batch", batch)
        lr0 = cfg.get("lr0", lr0)
        lrf = cfg.get("lrf", lrf)
        augment = cfg.get("augment", augment)
        mosaic = cfg.get("mosaic", mosaic)
        mixup = cfg.get("mixup", mixup)
        hsv_h = cfg.get("hsv_h", hsv_h)
        hsv_s = cfg.get("hsv_s", hsv_s)
        hsv_v = cfg.get("hsv_v", hsv_v)
        flipud = cfg.get("flipud", flipud)
        fliplr = cfg.get("fliplr", fliplr)
        run_name = cfg.get("run_name", run_name)
        register_model = cfg.get("register_model", register_model)

    # --- Connect to MLflow ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Base model: {base_model}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
    print(f"LR: {lr0} → {lrf}, Augment: {augment}")
    print()

    # --- Start MLflow run ---
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

        # Log all parameters
        mlflow.log_params({
            "base_model": base_model,
            "data_yaml": data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "lrf": lrf,
            "augment": augment,
            "mosaic": mosaic,
            "mixup": mixup,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "flipud": flipud,
            "fliplr": fliplr,
        })

        # --- Train YOLO ---
        model = YOLO(base_model)

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            lrf=lrf,
            augment=augment,
            mosaic=mosaic,
            mixup=mixup,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            flipud=flipud,
            fliplr=fliplr,
            project="runs/train",
            name=run_id,  # use MLflow run ID as folder name
            exist_ok=True,
            verbose=True,
        )

        # --- Log metrics ---
        # YOLO stores results in results.results_dict
        metrics = {}

        # YOLO may nest the output path (e.g. runs/detect/runs/train/{id})
        # so read the actual save_dir from the results object
        if hasattr(results, "save_dir"):
            results_dir = Path(results.save_dir)
        else:
            # Fallback: check both possible locations
            candidate = Path(f"runs/detect/runs/train/{run_id}")
            results_dir = candidate if candidate.exists() else Path(f"runs/train/{run_id}")

        print(f"Training output directory: {results_dir}")

        # Read metrics from results.csv that YOLO generates
        results_csv = results_dir / "results.csv"
        if results_csv.exists():
            import csv
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    # Get the last row (final epoch metrics)
                    last = rows[-1]
                    for key, value in last.items():
                        clean_key = key.strip().replace("/", "_").replace("(", "").replace(")", "")
                        try:
                            metrics[clean_key] = float(value.strip())
                        except (ValueError, AttributeError):
                            pass

                    # Log all epoch metrics for charting in MLflow UI
                    for i, row in enumerate(rows):
                        for key, value in row.items():
                            clean_key = key.strip().replace("/", "_").replace("(", "").replace(")", "")
                            try:
                                mlflow.log_metric(clean_key, float(value.strip()), step=i)
                            except (ValueError, AttributeError):
                                pass

        # Also try to get summary metrics from YOLO's results object
        if hasattr(results, "results_dict"):
            for key, value in results.results_dict.items():
                clean_key = key.replace("/", "_").replace("(", "").replace(")", "")
                if isinstance(value, (int, float)):
                    metrics[clean_key] = value

        if metrics:
            mlflow.log_metrics(metrics)
            print(f"\nLogged {len(metrics)} metrics to MLflow")

            # Print key metrics
            for key in ["metrics_mAP50B", "metrics_mAP50-95B", "metrics_precisionB", "metrics_recallB"]:
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")

        # --- Log the trained model ---
        best_model_path = results_dir / "weights" / "best.pt"
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="weights")
            print(f"\nModel artifact logged: {best_model_path}")
        else:
            print(f"\nWARNING: best.pt not found at {best_model_path}")

        # Log training plots (confusion matrix, results plots, etc.)
        for plot_file in results_dir.glob("*.png"):
            mlflow.log_artifact(str(plot_file), artifact_path="plots")

        # Log the data.yaml for reproducibility
        if os.path.exists(data_yaml):
            mlflow.log_artifact(data_yaml, artifact_path="config")

        # --- Register model if requested ---
        if register_model and best_model_path.exists():
            model_uri = f"runs:/{run_id}/weights/best.pt"
            registered = mlflow.register_model(model_uri, "autovision-traffic")
            print(f"\nModel registered: autovision-traffic v{registered.version}")
            print("You can now load it in the API with:")
            print(f"  MODEL_SOURCE=mlflow://autovision-traffic/latest")

        print(f"\nTraining complete! View results at:")
        print(f"  MLflow UI: {MLFLOW_TRACKING_URI}")
        print(f"  Run ID: {run_id}")

        return run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO with MLflow tracking")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--data-yaml", default="data/traffic/data.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--run-name", default=None, help="Name for this MLflow run")
    parser.add_argument("--register", action="store_true", help="Register model in MLflow")
    parser.add_argument("--config", default=None, help="YAML config file path")
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        augment=args.augment,
        mosaic=args.mosaic,
        mixup=args.mixup,
        run_name=args.run_name,
        register_model=args.register,
        config=args.config,
    )
