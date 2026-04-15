# AutoVision API

Traffic and vehicle detection API powered by YOLOv8 and FastAPI.
Supports single images, video files, and real-time webcam streaming.
Fine-tuned on traffic data with MLflow experiment tracking, tested with pytest, deployed to AWS.

> **Full project documentation with screenshots and architecture diagrams:**
> open [`DOCUMENTATION.html`](./DOCUMENTATION.html) in your browser
> (Make sure that the HTML file and the docs folder are in the same directory).

---

## Roadmap

| Phase | Status | What |
|-------|--------|------|
| 1 | ✅ Done | FastAPI service + Docker + video processing + webcam WebSocket stream |
| 2 | ✅ Done | YOLOv8 fine-tuning on Roboflow dataset + MLflow experiment tracking |
| 3 | ✅ Done | pytest test suite (32 tests) + GitHub Actions CI/CD + GHCR push |
| 4 | ✅ Done | AWS deployment — ECR container registry + EC2 compute |

---

## Quick Start

```bash
docker-compose up --build

# API docs:   http://localhost:8000/docs
# Live demo:  http://localhost:8000/stream
# MLflow UI:  http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check — model loaded status |
| `POST` | `/predict` | Upload image (JPEG/PNG/WebP) → annotated JPEG with bounding boxes |
| `POST` | `/predict/video` | Upload video (MP4/AVI/WebM/MOV) → annotated MP4 |
| `GET` | `/stream` | Live webcam detection in browser |
| `WS` | `/ws/detect` | WebSocket — real-time frame detection |

Detection results are returned in the response body (JPEG/MP4) **plus** a JSON header:
- `POST /predict` → `X-Detections` header
- `POST /predict/video` → `X-Summary` header

---

## Model Modes

The API supports two model sources, switched via environment variable:

```bash
# Pretrained YOLOv8n (COCO) — 8 vehicle-relevant classes
MODEL_SOURCE=local   # default

# Fine-tuned model from MLflow registry — 3 traffic classes (car / truck / bus)
MODEL_SOURCE=mlflow
```

---

## Phase 2 — Fine-Tuning with MLflow

### 1. Install training dependencies

```bash
pip install -r requirements-training.txt
```

### 2. Start MLflow tracking server

```bash
docker-compose up mlflow
# MLflow UI → http://localhost:5000
```

### 3. Download the dataset

```bash
python training/download_dataset.py --api-key YOUR_ROBOFLOW_KEY
```

### 4. Run experiments

```bash
python training/train.py --config training/configs/exp1_baseline.yaml   # 416px, default LR
python training/train.py --config training/configs/exp2_high_res.yaml   # 640px resolution
python training/train.py --config training/configs/exp3_heavy_aug.yaml  # heavy augmentation
```

### 5. Register the best model

```bash
python training/register_best.py
# Compares all runs by mAP50, registers best → Production in MLflow registry
```

### 6. Switch API to fine-tuned model

In `docker-compose.yml`:
```yaml
- MODEL_SOURCE=mlflow
```

Then rebuild: `docker-compose up --build`

---

## Phase 3 — Tests & CI/CD

```bash
pip install -r requirements-test.txt
pytest --cov=app -v
```

GitHub Actions pipeline on every push to `main` / `develop`:

1. **Lint** — `ruff check app/ tests/`
2. **Tests** — pytest on Python 3.10 and 3.11 (matrix)
3. **Docker build** — validates the Dockerfile builds cleanly
4. **Push to GHCR** — publishes image on `main` branch only

---

## Phase 4 — AWS Deployment

```bash
# Requires: AWS CLI configured + AWS_ACCOUNT_ID env var set
export AWS_ACCOUNT_ID=your_account_id
./scripts/deploy.sh
```

This builds the image, pushes it to ECR (eu-central-1), then on EC2:

```bash
docker pull <ECR_URI>/autovision-api:latest
docker run -d --name autovision -p 8000:8000 <ECR_URI>/autovision-api:latest
```

---

## Project Structure

```
autovision-api/
├── app/
│   ├── main.py              # FastAPI — health, predict, video endpoints
│   ├── model.py             # VehicleDetector — local + MLflow model loading
│   ├── schemas.py           # Pydantic response models
│   ├── stream.py            # WebSocket + /stream router
│   └── templates/
│       └── stream.html      # Webcam detection UI
├── training/
│   ├── download_dataset.py  # Pull dataset from Roboflow
│   ├── train.py             # Train YOLO + MLflow tracking
│   ├── register_best.py     # Compare runs, register best → Production
│   └── configs/
│       ├── exp1_baseline.yaml
│       ├── exp2_high_res.yaml
│       └── exp3_heavy_aug.yaml
├── tests/
│   └── test_api.py          # 32 tests across 4 classes (all mocked)
├── scripts/
│   └── deploy.sh            # Build + push to AWS ECR
├── docs/
│   └── img/                 # Project screenshots (embedded in DOCUMENTATION.html)
├── .github/
│   └── workflows/
│       └── ci.yml           # lint → test → docker-build → push pipeline
├── DOCUMENTATION.html       # Full project docs with diagrams + screenshots
├── Dockerfile               # Multi-stage build (builder + runtime)
├── docker-compose.yml       # api + mlflow services
├── requirements.txt
└── requirements-training.txt
```
