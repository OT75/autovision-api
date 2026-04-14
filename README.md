# AutoVision API

Traffic and vehicle detection API powered by YOLOv8 and FastAPI. Supports single images, video files, and real-time webcam streaming. Fine-tuned on traffic data with MLflow experiment tracking.

## Quick start

```bash
docker-compose up --build

# API:        http://localhost:8000/docs
# Live demo:  http://localhost:8000/stream
# MLflow UI:  http://localhost:5000
```

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Service health check |
| `POST /predict` | Upload image → annotated JPEG with bounding boxes |
| `POST /predict/video` | Upload video → annotated MP4 with bounding boxes |
| `GET /stream` | Live webcam detection in browser |

## Phase 2: Fine-tuning + MLflow

### Step 1 — Install training dependencies (locally, not Docker)

```bash
pip install -r requirements-training.txt
```

### Step 2 — Start MLflow

```bash
docker-compose up mlflow
# MLflow UI at http://localhost:5000
```

### Step 3 — Download the dataset

```bash
python training/download_dataset.py --api-key YOUR_ROBOFLOW_KEY
```

### Step 4 — Run experiments

```bash
# Experiment 1: baseline (416px, default LR)
python training/train.py --config training/configs/exp1_baseline.yaml

# Experiment 2: higher resolution (640px)
python training/train.py --config training/configs/exp2_high_res.yaml

# Experiment 3: aggressive augmentation + lower LR
python training/train.py --config training/configs/exp3_heavy_aug.yaml
```

### Step 5 — Compare in MLflow UI

Open http://localhost:5000, click the experiment, select all runs, click "Compare". You'll see mAP, precision, recall side by side.

### Step 6 — Register the best model

```bash
python training/register_best.py
```

### Step 7 — Switch API to use your fine-tuned model

In `docker-compose.yml`, change:
```yaml
- MODEL_SOURCE=mlflow
```

Then rebuild: `docker-compose up --build`

## Project structure

```
autovision-api/
├── app/
│   ├── main.py              # FastAPI — health, predict, video
│   ├── model.py             # YOLO wrapper (local + MLflow loading)
│   ├── schemas.py           # Pydantic models
│   ├── stream.py            # WebSocket + webcam router
│   └── templates/
│       └── stream.html      # Webcam UI
├── training/
│   ├── download_dataset.py  # Pull dataset from Roboflow
│   ├── train.py             # Train YOLO with MLflow tracking
│   ├── register_best.py     # Compare runs, register best model
│   └── configs/
│       ├── exp1_baseline.yaml
│       ├── exp2_high_res.yaml
│       └── exp3_heavy_aug.yaml
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml       # API + MLflow
├── requirements.txt         # API dependencies
└── requirements-training.txt # Training dependencies (local only)
```

## Roadmap

| Phase | Status | What |
|-------|--------|------|
| 1 | Done | API + Docker + video + webcam stream |
| 2 | Done | Fine-tune YOLO + MLflow experiment tracking |
| 3 | Done | pytest + GitHub Actions CI/CD |
| 4 | Done | AWS deployment (ECR, EC2, S3) |
| 5 | — | Prometheus + Grafana monitoring |
| 6 | — | Gradio demo + blog post |
