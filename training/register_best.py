"""
Compare all MLflow runs and register the best model.

Usage:
    python training/register_best.py

This script:
1. Fetches all runs from the autovision experiment
2. Finds the one with the highest mAP50
3. Registers it in MLflow Model Registry as "autovision-traffic"
"""

import os

import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "autovision-traffic-detection"
MODEL_NAME = "autovision-traffic"


def find_map50_key(metrics: dict) -> tuple[str, float]:
    """
    Find the mAP50 metric regardless of how YOLO named it.
    
    Different YOLO versions log metrics under different names:
    - metrics/mAP50(B)
    - metrics_mAP50B
    - metrics_mAP50(B)
    - val/mAP50
    
    This function tries all common patterns.
    """
    # Patterns to look for (ordered by priority)
    map50_patterns = ["mAP50(B)", "mAP50B", "mAP50", "map50", "val_mAP50"]
    
    for key, value in metrics.items():
        key_lower = key.lower().replace("/", "_").replace(" ", "")
        for pattern in map50_patterns:
            if pattern.lower() in key_lower and "95" not in key_lower:
                return key, float(value)
    
    return None, 0.0


def find_metric(metrics: dict, search_terms: list[str]) -> tuple[str, float]:
    """Find a metric by searching for partial matches."""
    for key, value in metrics.items():
        key_lower = key.lower().replace("/", "_").replace(" ", "")
        for term in search_terms:
            if term.lower() in key_lower:
                return key, float(value)
    return None, 0.0


def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
        print("Have you run any training yet?")
        return

    # Get all completed runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=20,
    )

    if not runs:
        print("No completed runs found.")
        return

    # First, discover metric names from the first run
    sample_metrics = runs[0].data.metrics
    print("Available metrics in your runs:")
    for key in sorted(sample_metrics.keys()):
        print(f"  {key}: {sample_metrics[key]:.4f}")
    print()

    # Find and display all runs with their metrics
    print("=" * 80)
    print("All completed runs:")
    print("=" * 80)

    run_data = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        
        name = params.get("run_name", run.info.run_name or run.info.run_id[:12])
        _, map50 = find_map50_key(metrics)
        _, map5095 = find_metric(metrics, ["mAP50-95", "mAP5095"])
        _, precision = find_metric(metrics, ["precision"])
        _, recall = find_metric(metrics, ["recall"])
        
        run_data.append({
            "run": run,
            "name": name,
            "map50": map50,
            "map5095": map5095,
            "precision": precision,
            "recall": recall,
        })

    # Sort by mAP50 descending
    run_data.sort(key=lambda r: r["map50"], reverse=True)

    print(f"{'Run Name':<25} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 80)
    for rd in run_data:
        print(f"{rd['name']:<25} {rd['map50']:>8.4f} {rd['map5095']:>10.4f} {rd['precision']:>10.4f} {rd['recall']:>8.4f}")

    # Best run
    best = run_data[0]
    best_run = best["run"]

    print(f"\nBest run: {best['name']} (mAP50: {best['map50']:.4f})")

    # Check if model artifact exists
    artifacts = client.list_artifacts(best_run.info.run_id)
    artifact_names = [a.path for a in artifacts]
    print(f"Artifacts: {artifact_names}")

    model_path = None

    # Search strategy: check common locations for best.pt
    search_paths = [
        "model/best.pt",       # where train.py tries to log it
        "weights/best.pt",     # where YOLO puts it natively
        "best.pt",             # root level
    ]

    for path in search_paths:
        try:
            # Try to list the parent directory to see if the file exists
            parent = "/".join(path.split("/")[:-1]) if "/" in path else ""
            sub_artifacts = client.list_artifacts(best_run.info.run_id, parent) if parent else artifacts
            sub_names = [a.path for a in sub_artifacts]
            if path in sub_names:
                model_path = path
                print(f"Found model at: {model_path}")
                break
        except Exception:
            continue

    # If still not found, do a deep search
    if not model_path:
        for a in artifacts:
            if a.is_dir:
                try:
                    sub_artifacts = client.list_artifacts(best_run.info.run_id, a.path)
                    for sa in sub_artifacts:
                        if sa.path.endswith("best.pt"):
                            model_path = sa.path
                            print(f"Found model at: {model_path}")
                            break
                except Exception:
                    pass
            if model_path:
                break

    if not model_path:
        print("\nWarning: No model artifact (.pt file) found in the best run.")
        print("Make sure training completed successfully and the model was logged.")
        return

    # Register the model
    model_uri = f"runs:/{best_run.info.run_id}/{model_path}"
    print(f"\nRegistering model from: {model_uri}")

    try:
        result = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Model registered: {MODEL_NAME} v{result.version}")

        # Try to transition to Production
        try:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=result.version,
                stage="Production",
            )
            print(f"Model transitioned to Production stage")
        except Exception as e:
            print(f"Note: Could not set stage (this is OK): {e}")

    except Exception as e:
        print(f"Registration note: {e}")
        print("Model may already be registered. Check MLflow UI.")

    print(f"\nDone! To use this model in your API:")
    print(f"  1. In docker-compose.yml, set MODEL_SOURCE=mlflow")
    print(f"  2. Run: docker-compose up --build")


if __name__ == "__main__":
    register_best_model()
