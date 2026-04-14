"""
AutoVision API — Test Suite (Phase 3)

Patches:
- app.model.YOLO        — prevents weight download
- app.model.detector    — replaces the singleton with a pre-loaded mock
"""

import io
import json
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_jpeg(width=320, height=240) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def make_png(width=320, height=240) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_detector():
    """
    A fully mocked VehicleDetector that never touches YOLO or MLflow.
    Mirrors the real detector's interface exactly.
    """
    det = MagicMock()
    det.is_loaded = True
    det.model_name = "yolov8n.pt"

    # predict_annotated returns (jpeg_bytes, result_dict)
    fake_result = {
        "detections": [
            {"class_id": 2, "class_name": "car", "confidence": 0.91,
             "bbox": [10.0, 20.0, 100.0, 80.0]}
        ],
        "num_detections": 1,
        "image_width": 320,
        "image_height": 240,
        "model_name": "yolov8n.pt",
    }
    fake_jpeg = make_jpeg()
    det.predict_annotated.return_value = (fake_jpeg, fake_result)

    # process_video_annotated returns (mp4_bytes, result_dict)
    fake_video_result = {
        "video_info": {"width": 320, "height": 240, "fps": 30.0,
                       "total_frames": 90, "duration_sec": 3.0},
        "processing_info": {"sample_fps": 2, "frames_analyzed": 6,
                            "processing_time_sec": 1.2, "model_name": "yolov8n.pt"},
        "summary": {"total_detections": 3, "class_counts": {"car": 3}},
        "frames": [],
    }
    det.process_video_annotated.return_value = (b"fake_mp4_bytes", fake_video_result)

    return det


@pytest.fixture(scope="session")
def client(mock_detector):
    """
    TestClient with the detector singleton replaced.
    We patch app.model.detector (the module-level singleton) AND
    skip the lifespan loader so detector.load() is never called.
    """
    import os
    os.environ.setdefault("MODEL_SOURCE", "local")
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

    with patch("app.model.detector", mock_detector), \
         patch("app.main.detector", mock_detector):
        from app.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_is_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_model_loaded_true(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_model_name_present(self, client):
        assert "model_name" in client.get("/health").json()

    def test_content_type_json(self, client):
        assert "application/json" in client.get("/health").headers["content-type"]


# ---------------------------------------------------------------------------
# 2. POST /predict  (image)
# ---------------------------------------------------------------------------

class TestPredictImage:

    def test_jpeg_returns_200(self, client):
        resp = client.post("/predict",
                           files={"file": ("test.jpg", make_jpeg(), "image/jpeg")})
        assert resp.status_code == 200

    def test_response_is_jpeg(self, client):
        resp = client.post("/predict",
                           files={"file": ("test.jpg", make_jpeg(), "image/jpeg")})
        assert resp.headers["content-type"] == "image/jpeg"

    def test_response_is_valid_image(self, client):
        resp = client.post("/predict",
                           files={"file": ("test.jpg", make_jpeg(), "image/jpeg")})
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "JPEG"

    def test_x_detections_header_present(self, client):
        resp = client.post("/predict",
                           files={"file": ("test.jpg", make_jpeg(), "image/jpeg")})
        assert "x-detections" in resp.headers

    def test_x_detections_is_valid_json(self, client):
        resp = client.post("/predict",
                           files={"file": ("test.jpg", make_jpeg(), "image/jpeg")})
        data = json.loads(resp.headers["x-detections"])
        assert "detections" in data
        assert "num_detections" in data

    def test_png_accepted(self, client):
        resp = client.post("/predict",
                           files={"file": ("test.png", make_png(), "image/png")})
        assert resp.status_code == 200

    def test_wrong_type_returns_422(self, client):
        resp = client.post("/predict",
                           files={"file": ("test.txt", b"not an image", "text/plain")})
        assert resp.status_code == 422

    def test_empty_file_returns_422(self, client):
        resp = client.post("/predict",
                           files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert resp.status_code == 422

    def test_no_file_returns_422(self, client):
        assert client.post("/predict").status_code == 422

    def test_small_image(self, client):
        resp = client.post("/predict",
                           files={"file": ("tiny.jpg", make_jpeg(32, 32), "image/jpeg")})
        assert resp.status_code == 200

    def test_large_image(self, client):
        resp = client.post("/predict",
                           files={"file": ("hd.jpg", make_jpeg(1280, 720), "image/jpeg")})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. POST /predict/video
# ---------------------------------------------------------------------------

class TestPredictVideo:

    def _post(self, client, data=b"fake_mp4", ct="video/mp4", name="test.mp4"):
        return client.post("/predict/video",
                           files={"file": (name, data, ct)})

    def test_mp4_returns_200(self, client):
        assert self._post(client).status_code == 200

    def test_response_is_video_mp4(self, client):
        resp = self._post(client)
        assert resp.headers["content-type"] == "video/mp4"

    def test_x_summary_header_present(self, client):
        resp = self._post(client)
        assert "x-summary" in resp.headers

    def test_x_summary_has_expected_keys(self, client):
        resp = self._post(client)
        data = json.loads(resp.headers["x-summary"])
        assert "video_info" in data
        assert "processing_info" in data
        assert "summary" in data

    def test_wrong_type_returns_422(self, client):
        assert self._post(client, b"text", "text/plain", "test.txt").status_code == 422

    def test_empty_file_returns_422(self, client):
        assert self._post(client, b"", "video/mp4").status_code == 422

    def test_no_file_returns_422(self, client):
        assert client.post("/predict/video").status_code == 422

    def test_sample_fps_param_accepted(self, client):
        resp = client.post("/predict/video?sample_fps=5",
                           files={"file": ("test.mp4", b"fake_mp4", "video/mp4")})
        assert resp.status_code == 200

    def test_sample_fps_out_of_range_returns_422(self, client):
        resp = client.post("/predict/video?sample_fps=999",
                           files={"file": ("test.mp4", b"fake_mp4", "video/mp4")})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. API structure
# ---------------------------------------------------------------------------

class TestApiStructure:

    def test_openapi_available(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_docs_available(self, client):
        assert client.get("/docs").status_code == 200

    def test_unknown_route_404(self, client):
        assert client.get("/nonexistent").status_code == 404

    def test_predict_in_schema(self, client):
        assert "/predict" in client.get("/openapi.json").json()["paths"]

    def test_predict_video_in_schema(self, client):
        assert "/predict/video" in client.get("/openapi.json").json()["paths"]

    def test_health_in_schema(self, client):
        assert "/health" in client.get("/openapi.json").json()["paths"]

    def test_stream_in_schema(self, client):
        assert "/stream" in client.get("/openapi.json").json()["paths"]