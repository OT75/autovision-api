# AutoVision API

Traffic and vehicle detection API built with YOLOv8 and FastAPI. Detects cars, trucks, buses, motorcycles, bicycles, pedestrians, traffic lights, and stop signs.

**This is a living MLOps portfolio project.** Each phase adds production tools:

| Phase | What | Tools |
|-------|------|-------|
| **1 (current)** | Minimal working API | FastAPI, Docker, YOLOv8 |
| 2 | Experiment tracking | MLflow, fine-tuning |
| 3 | Testing + CI/CD | pytest, GitHub Actions |
| 4 | Cloud deployment | AWS ECR, EC2, S3 |
| 5 | Monitoring + drift | Prometheus, Grafana, Evidently |
| 6 | Ship it | Gradio demo, blog post |

## Quick start

```bash
# Build and run with Docker (you already know this part)
docker-compose up --build

# The API is now at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Endpoints

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "model_loaded": true, "model_name": "yolov8n.pt"}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict -F "file=@traffic_photo.jpg"
```
```json
{
  "success": true,
  "num_detections": 5,
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.9234,
      "bbox": [120.5, 200.3, 450.1, 380.7]
    }
  ],
  "image_width": 1280,
  "image_height": 720,
  "model_name": "yolov8n.pt"
}
```

### `POST /predict/video`
Upload a video file for frame-by-frame analysis. You can optionally specify a `sample_fps` parameter to speed up processing (e.g. process just 2 frames per second).

```bash
curl -X POST http://localhost:8000/predict/video \
  -F "file=@traffic_video.mp4" \
  -F "sample_fps=2"
```

```json
{
  "success": true,
  "video_info": {
    "width": 1280,
    "height": 720,
    "fps": 30.0,
    "total_frames": 300,
    "duration_sec": 10.0
  },
  "processing_info": {
    "sample_fps": 2,
    "frames_analyzed": 20,
    "processing_time_sec": 1.5,
    "model_name": "yolov8n.pt"
  },
  "summary": {
    "total_detections": 45,
    "class_counts": {
      "car": 40,
      "person": 5
    }
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "num_detections": 2,
      "detections": [
        {
          "class_id": 2,
          "class_name": "car",
          "confidence": 0.912,
          "bbox": [10.5, 20.3, 150.1, 80.7]
        }
      ]
    }
  ]
}
```

### `GET /stream` (Live Webcam)
Open the app in your browser at `http://localhost:8000/stream` to view a live webcam interface.

This page connects to your webcam, streams frames to the server over a WebSocket connection (`/ws/detect`), runs the YOLO inference, and returns visually annotated frames displaying bounding boxes in real-time.

## Run without Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v
```

## Project structure

```
autovision-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, routes, lifespan
│   ├── model.py         # YOLO model loading + inference
│   └── schemas.py       # Pydantic request/response models
├── tests/
│   └── test_api.py      # Endpoint tests
├── scripts/
│   └── test_locally.sh  # Manual curl test script
├── Dockerfile           # Multi-stage build
├── docker-compose.yml   # Run everything
├── requirements.txt     # Python dependencies
└── README.md
```
