"""
Compare all MLflow runs and register the best model in the Model Registry.

Usage:
    python training/register_best.py

What it does:
1. Finds all finished runs in the autovision experiment
2. Picks the one with the highest mAP50
3. Registers it as "autovision-traffic" in MLflow Model Registry → Production stage

Run this once after your training experiments are done.
Then set MODEL_SOURCE=mlflow in docker-compose.yml and rebuild.
"""

import os

import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "autovision-traffic-detection"
MODEL_NAME = "autovision-traffic"


def register_best():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found. Run training first.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=20,
    )

    if not runs:
        print("No finished runs found.")
        return

    # Sort runs by mAP50 (the metric name we log in train.py)
    def get_map50(run):
        return run.data.metrics.get("mAP50", 0.0)

    runs.sort(key=get_map50, reverse=True)

    # Print comparison table
    print(f"\n{'Run':<20} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)
    for run in runs:
        m = run.data.metrics
        name = run.data.params.get("run_name", run.info.run_id[:8])
        print(
            f"{name:<20} "
            f"{m.get('mAP50', 0):.4f}   "
            f"{m.get('mAP50-95', 0):>10.4f}   "
            f"{m.get('precision', 0):>10.4f}   "
            f"{m.get('recall', 0):>8.4f}"
        )

    best_run = runs[0]
    best_name = best_run.data.params.get("run_name", best_run.info.run_id[:8])
    best_map50 = get_map50(best_run)
    print(f"\nBest run: {best_name}  (mAP50: {best_map50:.4f})")

    # The artifact path is always weights/best.pt — that's what train.py logs
    model_uri = f"runs:/{best_run.info.run_id}/weights/best.pt"
    print(f"Registering: {model_uri}")

    try:
        result = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Registered '{MODEL_NAME}' v{result.version}")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Production",
        )
        print(f"Staged to Production")

    except Exception as e:
        print(f"Note: {e}")
        print("Model may already be registered — check MLflow UI at http://localhost:5000")

    print(f"\nNext step: set MODEL_SOURCE=mlflow in docker-compose.yml, then docker-compose up --build")


if __name__ == "__main__":
    register_best()
