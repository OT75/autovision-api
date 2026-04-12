"""
AutoVision API — Tests

Covers:
- GET  /health
- POST /predict (image)
- POST /predict/video
- GET  /stream (webcam page)

Run with: pytest tests/ -v
"""

import io
import tempfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ================================================================
# Helpers
# ================================================================

def make_test_image(width=640, height=480) -> io.BytesIO:
    """Create a minimal valid JPEG for testing."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def make_test_video(width=320, height=240, fps=10, frames=20) -> io.BytesIO:
    """Create a minimal valid MP4 video for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))

        for i in range(frames):
            # Create a frame with a moving rectangle (gives YOLO something to look at)
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = int((i / frames) * (width - 60))
            cv2.rectangle(frame, (x, 80), (x + 60, 180), (0, 255, 0), -1)
            writer.write(frame)

        writer.release()

        # Read the file back into bytes
        with open(tmp.name, "rb") as f:
            video_bytes = f.read()

    buf = io.BytesIO(video_bytes)
    buf.seek(0)
    return buf


# ================================================================
# Health
# ================================================================

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_shows_model_loaded():
    response = client.get("/health")
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"


# ================================================================
# Image prediction
# ================================================================

def test_predict_rejects_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_rejects_wrong_file_type():
    fake = io.BytesIO(b"not an image")
    response = client.post("/predict", files={"file": ("test.txt", fake, "text/plain")})
    assert response.status_code == 422


def test_predict_rejects_empty_file():
    empty = io.BytesIO(b"")
    response = client.post("/predict", files={"file": ("e.jpg", empty, "image/jpeg")})
    assert response.status_code == 422


def test_predict_accepts_valid_image():
    buf = make_test_image()
    response = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert isinstance(data["detections"], list)
    assert data["image_width"] == 640
    assert data["image_height"] == 480
    assert "model_name" in data


def test_predict_detection_schema():
    buf = make_test_image()
    response = client.post("/predict", files={"file": ("t.jpg", buf, "image/jpeg")})
    data = response.json()

    for det in data["detections"]:
        assert isinstance(det["class_id"], int)
        assert isinstance(det["class_name"], str)
        assert 0.0 <= det["confidence"] <= 1.0
        assert len(det["bbox"]) == 4


# ================================================================
# Video prediction
# ================================================================

def test_video_rejects_wrong_file_type():
    fake = io.BytesIO(b"not a video")
    response = client.post(
        "/predict/video",
        files={"file": ("test.txt", fake, "text/plain")},
    )
    assert response.status_code == 422


def test_video_rejects_empty_file():
    empty = io.BytesIO(b"")
    response = client.post(
        "/predict/video",
        files={"file": ("e.mp4", empty, "video/mp4")},
    )
    assert response.status_code == 422


def test_video_accepts_valid_video():
    """Upload a synthetic video and check the response structure."""
    buf = make_test_video()
    response = client.post(
        "/predict/video",
        files={"file": ("test.mp4", buf, "video/mp4")},
        data={"sample_fps": "2"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True

    # Check video_info
    assert data["video_info"]["width"] == 320
    assert data["video_info"]["height"] == 240
    assert data["video_info"]["total_frames"] == 20

    # Check processing_info
    assert data["processing_info"]["sample_fps"] == 2
    assert data["processing_info"]["frames_analyzed"] >= 1
    assert "processing_time_sec" in data["processing_info"]

    # Check summary
    assert "total_detections" in data["summary"]
    assert isinstance(data["summary"]["class_counts"], dict)

    # Check frames
    assert isinstance(data["frames"], list)
    assert len(data["frames"]) >= 1

    for frame in data["frames"]:
        assert "frame_index" in frame
        assert "timestamp_sec" in frame
        assert isinstance(frame["detections"], list)


def test_video_sample_fps_parameter():
    """Higher sample_fps should analyze more frames."""
    buf1 = make_test_video(frames=30)
    response1 = client.post(
        "/predict/video",
        files={"file": ("t.mp4", buf1, "video/mp4")},
        data={"sample_fps": "1"},
    )

    buf2 = make_test_video(frames=30)
    response2 = client.post(
        "/predict/video",
        files={"file": ("t.mp4", buf2, "video/mp4")},
        data={"sample_fps": "5"},
    )

    frames1 = response1.json()["processing_info"]["frames_analyzed"]
    frames2 = response2.json()["processing_info"]["frames_analyzed"]

    # sample_fps=5 should analyze more frames than sample_fps=1
    assert frames2 > frames1


# ================================================================
# Webcam stream page
# ================================================================

def test_stream_page_returns_html():
    response = client.get("/stream")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AutoVision" in response.text
    assert "WebSocket" in response.text
