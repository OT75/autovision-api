"""
YOLO model wrapper for AutoVision API.

MODEL_SOURCE=local  (default) — loads yolov8n.pt
MODEL_SOURCE=mlflow            — loads the Production model from MLflow registry
"""

import io
import os
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# --- Config ---
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "autovision-traffic")
MLFLOW_ARTIFACTS_DIR = "/mlflow/artifacts"

# Pretrained YOLOv8 (COCO) — vehicle-relevant classes only
VEHICLE_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign",
}

# Fine-tuned model classes (Roboflow dataset: 3 classes, IDs 0–2)
FINETUNED_CLASSES = {0: "car", 1: "truck", 2: "bus"}

# BGR colors for bounding boxes
CLASS_COLORS = {
    0: (0, 255, 128), 1: (255, 200, 0), 2: (255, 100, 0),
    3: (0, 165, 255), 5: (0, 255, 255), 7: (128, 0, 255),
    9: (0, 0, 255),  11: (0, 0, 200),
}
FINETUNED_CLASS_COLORS = {
    0: (255, 100, 0), 1: (128, 0, 255), 2: (0, 255, 255),
}


# --- Helpers ---

def _extract_detections(boxes, class_map: dict) -> list[dict]:
    """Parse YOLO result boxes into a clean list of dicts."""
    detections = []
    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        if class_id not in class_map:
            continue
        detections.append({
            "class_id": class_id,
            "class_name": class_map[class_id],
            "confidence": round(float(boxes.conf[i].item()), 4),
            "bbox": [round(c, 2) for c in boxes.xyxy[i].tolist()],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_detections_on_frame(
    frame: np.ndarray,
    detections: list[dict],
    color_map: dict = CLASS_COLORS,
) -> np.ndarray:
    """Draw bounding boxes + labels onto a BGR frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        color = color_map.get(det["class_id"], (200, 200, 200))
        label = f"{det['class_name']} {det['confidence']:.0%}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


# --- Detector ---

class VehicleDetector:
    """Wraps YOLO for image, video, and WebSocket frame inference."""

    def __init__(self):
        self.model = None
        self.model_name = MODEL_NAME
        self._class_map = VEHICLE_CLASSES
        self._color_map = CLASS_COLORS

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self):
        print(f"[AutoVision] MODEL_SOURCE={MODEL_SOURCE}", flush=True)
        if MODEL_SOURCE == "mlflow":
            self._load_from_mlflow()
        else:
            self._load_local()
        print(f"[AutoVision] Ready — model={self.model_name}", flush=True)

    def _load_local(self):
        print(f"[AutoVision] Loading local model: {MODEL_NAME}", flush=True)
        self.model = YOLO(MODEL_NAME)

    def _load_from_mlflow(self):
        """
        Load the Production model from MLflow registry.
        Reads best.pt directly from the shared Docker volume (/mlflow/artifacts)
        to avoid download_artifacts cross-container issues.
        Falls back to local if anything goes wrong.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
            if not versions:
                versions = client.get_latest_versions(MLFLOW_MODEL_NAME)

            if not versions:
                print(f"[AutoVision] No registered model '{MLFLOW_MODEL_NAME}'. Falling back to local.", flush=True)
                self._load_local()
                return

            version = versions[0]
            run_id = version.run_id
            experiment_id = client.get_run(run_id).info.experiment_id

            # Check all known artifact layout patterns (upload location varies
            # depending on whether artifact_uri was set correctly at experiment creation)
            candidates = [
                f"{MLFLOW_ARTIFACTS_DIR}/{experiment_id}/{run_id}/artifacts/weights/best.pt",
                f"{MLFLOW_ARTIFACTS_DIR}/{run_id}/artifacts/weights/best.pt",
                f"{MLFLOW_ARTIFACTS_DIR}/weights/best.pt",
                f"{MLFLOW_ARTIFACTS_DIR}/best.pt",
            ]
            best_pt = next((p for p in candidates if os.path.exists(p)), None)

            if best_pt is None:
                print(f"[AutoVision] Model not found. Checked:", flush=True)
                for p in candidates:
                    print(f"[AutoVision]   {p}", flush=True)
                print(f"[AutoVision] Falling back to local.", flush=True)
                self._load_local()
                return

            size_mb = os.path.getsize(best_pt) / 1024 / 1024
            print(f"[AutoVision] Loading MLflow model ({size_mb:.1f} MB): {best_pt}", flush=True)

            self.model = YOLO(best_pt)
            self.model_name = f"mlflow://{MLFLOW_MODEL_NAME}/v{version.version}"
            self._class_map = FINETUNED_CLASSES
            self._color_map = FINETUNED_CLASS_COLORS

        except Exception as exc:
            print(f"[AutoVision] MLflow load failed: {exc}. Falling back to local.", flush=True)
            self._load_local()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_annotated(self, image_bytes: bytes) -> tuple[bytes, dict]:
        """Image bytes → annotated JPEG bytes + detection dict."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        image_np = np.array(Image.open(io.BytesIO(image_bytes)))
        result = self.model(source=image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        img_h, img_w = result.orig_shape

        detections = _extract_detections(result.boxes, self._class_map)
        annotated_bgr = draw_detections_on_frame(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), detections, self._color_map
        )
        _, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        return buf.tobytes(), {
            "detections": detections,
            "num_detections": len(detections),
            "image_width": img_w,
            "image_height": img_h,
            "model_name": self.model_name,
        }

    def process_video_annotated(self, video_bytes: bytes, sample_fps: int = 2) -> tuple[bytes, dict]:
        """Video bytes → annotated MP4 bytes + summary dict."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            input_path = tmp.name
        output_path = input_path.replace(".mp4", "_annotated.mp4")

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_interval = max(1, int(video_fps / sample_fps))

            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height)
            )

            frames_data, current_detections = [], []
            class_counts: dict[str, int] = {}
            total_detections = 0
            frame_idx = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    results = self.model(
                        source=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        conf=CONFIDENCE_THRESHOLD, verbose=False
                    )
                    current_detections = _extract_detections(results[0].boxes, self._class_map)
                    frames_data.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": round(frame_idx / video_fps, 2),
                        "num_detections": len(current_detections),
                        "detections": current_detections,
                    })
                    total_detections += len(current_detections)
                    for det in current_detections:
                        class_counts[det["class_name"]] = class_counts.get(det["class_name"], 0) + 1

                writer.write(draw_detections_on_frame(frame, current_detections, self._color_map))
                frame_idx += 1

            cap.release()
            writer.release()

            with open(output_path, "rb") as f:
                annotated_video_bytes = f.read()

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

        return annotated_video_bytes, {
            "video_info": {
                "width": width, "height": height,
                "fps": round(video_fps, 1),
                "total_frames": total_frames,
                "duration_sec": round(total_frames / video_fps, 2),
            },
            "processing_info": {
                "sample_fps": sample_fps,
                "frames_analyzed": len(frames_data),
                "processing_time_sec": round(time.time() - start_time, 2),
                "model_name": self.model_name,
            },
            "summary": {
                "total_detections": total_detections,
                "class_counts": class_counts,
            },
            "frames": frames_data,
        }

    def predict_frame(self, frame: np.ndarray) -> list[dict]:
        """Single numpy frame → detections list (used by WebSocket)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        results = self.model(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        return _extract_detections(results[0].boxes, self._class_map)


# Singleton used by main.py and stream.py
detector = VehicleDetector()