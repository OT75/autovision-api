"""
YOLO model wrapper for AutoVision API.

Model loading supports two modes:
- Local file: MODEL_SOURCE=local (default) — loads from MODEL_NAME file
- MLflow:     MODEL_SOURCE=mlflow — loads the best model from MLflow registry

Supports:
- Single image prediction (POST /predict)
- Video file processing frame-by-frame (POST /predict/video)
- Real-time webcam frames via WebSocket (/ws/detect)
"""

import io
import os
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# --- Configuration ---
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")  # "local" or "mlflow"
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# MLflow settings (only used when MODEL_SOURCE=mlflow)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "autovision-traffic")

# Shared volume where MLflow stores artifacts (mounted in docker-compose)
MLFLOW_ARTIFACTS_DIR = "/mlflow/artifacts"

# Vehicle-related COCO class IDs (pretrained model)
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
    0: "person",
}

# Fine-tuned model classes — sequential IDs 0/1/2 (NOT COCO IDs)
FINETUNED_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
}

# Colors for COCO classes (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 128),     # person — green
    1: (255, 200, 0),     # bicycle — cyan
    2: (255, 100, 0),     # car — blue
    3: (0, 165, 255),     # motorcycle — orange
    5: (0, 255, 255),     # bus — yellow
    7: (128, 0, 255),     # truck — purple
    9: (0, 0, 255),       # traffic light — red
    11: (0, 0, 200),      # stop sign — dark red
}

# Colors for fine-tuned model
FINETUNED_CLASS_COLORS = {
    0: (255, 100, 0),     # car — blue
    1: (128, 0, 255),     # truck — purple
    2: (0, 255, 255),     # bus — yellow
}


def _extract_detections(boxes, class_map: dict) -> list[dict]:
    """Extract and filter detections from YOLO result boxes."""
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
    frame: np.ndarray, detections: list[dict], color_map: dict = CLASS_COLORS
) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        class_id = det["class_id"]
        label = f"{det['class_name']} {det['confidence']:.0%}"
        color = color_map.get(class_id, (200, 200, 200))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 4, y1),
            color, -1,
        )
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


class VehicleDetector:
    """Wraps YOLO model for vehicle/traffic detection."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._class_map = VEHICLE_CLASSES
        self._color_map = CLASS_COLORS

    def load(self):
        """Load the YOLO model from local file or MLflow registry."""
        # Force print flushing so Docker logs show immediately
        print(f"[AutoVision] MODEL_SOURCE={MODEL_SOURCE}", flush=True)

        if MODEL_SOURCE == "mlflow":
            self._load_from_mlflow()
        else:
            self._load_local()

        print(f"[AutoVision] Final state: loaded={self.is_loaded}, name={self.model_name}", flush=True)

    def _load_local(self):
        """Load model from a local file."""
        print(f"[AutoVision] Loading local model: {self.model_name}", flush=True)
        self.model = YOLO(self.model_name)
        print(f"[AutoVision] Local model loaded: {self.model_name}", flush=True)

    def _load_from_mlflow(self):
        """
        Load the production model from MLflow Model Registry.
        
        Strategy: use MLflow API to find the run ID of the registered model,
        then read the best.pt file directly from the shared Docker volume
        at /mlflow/artifacts/. This avoids the download_artifacts API which
        doesn't work across containers.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            print(f"[AutoVision] Connecting to MLflow at {MLFLOW_TRACKING_URI}...", flush=True)
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient()

            # Get the latest Production version
            versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
            if not versions:
                versions = client.get_latest_versions(MLFLOW_MODEL_NAME)
            
            if not versions:
                print(f"[AutoVision] No model versions in registry '{MLFLOW_MODEL_NAME}'.", flush=True)
                print(f"[AutoVision] Falling back to local model.", flush=True)
                self._load_local()
                return

            version = versions[0]
            run_id = version.run_id
            print(f"[AutoVision] Found: {MLFLOW_MODEL_NAME} v{version.version} (run: {run_id[:8]})", flush=True)

            # Get the experiment ID to build the artifact path
            run_info = client.get_run(run_id)
            experiment_id = run_info.info.experiment_id

            # Build the direct filesystem path to the artifact
            # MLflow stores artifacts at: {artifact_root}/{experiment_id}/{run_id}/artifacts/
            best_pt = None
            search_patterns = [
                f"{MLFLOW_ARTIFACTS_DIR}/best.pt",
                f"{MLFLOW_ARTIFACTS_DIR}/{experiment_id}/{run_id}/artifacts/weights/best.pt",
                f"{MLFLOW_ARTIFACTS_DIR}/{experiment_id}/{run_id}/artifacts/model/best.pt",
                f"{MLFLOW_ARTIFACTS_DIR}/{experiment_id}/{run_id}/artifacts/best.pt",
            ]

            for pattern in search_patterns:
                print(f"[AutoVision] Checking: {pattern}", flush=True)
                if os.path.exists(pattern):
                    best_pt = pattern
                    print(f"[AutoVision] Found model at: {best_pt}", flush=True)
                    break

            # If exact paths didn't work, search recursively
            if best_pt is None:
                base_dir = f"{MLFLOW_ARTIFACTS_DIR}/{experiment_id}/{run_id}"
                print(f"[AutoVision] Searching recursively in: {base_dir}", flush=True)
                if os.path.exists(base_dir):
                    for root, dirs, files in os.walk(base_dir):
                        for f in files:
                            if f == "best.pt":
                                best_pt = os.path.join(root, f)
                                print(f"[AutoVision] Found model at: {best_pt}", flush=True)
                                break
                        if best_pt:
                            break
                    if not best_pt:
                        # Show what's actually there
                        print(f"[AutoVision] Contents of {base_dir}:", flush=True)
                        for root, dirs, files in os.walk(base_dir):
                            level = root.replace(base_dir, "").count(os.sep)
                            indent = "  " * level
                            print(f"[AutoVision]   {indent}{os.path.basename(root)}/", flush=True)
                            for f in files:
                                print(f"[AutoVision]   {indent}  {f}", flush=True)
                else:
                    # Show what experiment directories exist
                    print(f"[AutoVision] {MLFLOW_ARTIFACTS_DIR} contents:", flush=True)
                    if os.path.exists(MLFLOW_ARTIFACTS_DIR):
                        for item in os.listdir(MLFLOW_ARTIFACTS_DIR):
                            print(f"[AutoVision]   {item}", flush=True)
                    else:
                        print(f"[AutoVision]   Directory does not exist!", flush=True)

            if best_pt is None:
                print(f"[AutoVision] Could not find best.pt. Falling back to local.", flush=True)
                self._load_local()
                return

            # Load the model
            file_size = os.path.getsize(best_pt) / 1024 / 1024
            print(f"[AutoVision] Loading model ({file_size:.1f} MB)...", flush=True)

            self.model = YOLO(best_pt)
            self.model_name = f"mlflow://{MLFLOW_MODEL_NAME}/v{version.version}"
            self._class_map = FINETUNED_CLASSES
            self._color_map = FINETUNED_CLASS_COLORS
            print(f"[AutoVision] Model loaded from MLflow: {self.model_name}", flush=True)
            print(f"[AutoVision] Using fine-tuned classes: {self._class_map}", flush=True)

        except Exception as e:
            import traceback
            print(f"[AutoVision] MLflow loading failed: {e}", flush=True)
            traceback.print_exc()
            print(f"[AutoVision] Falling back to local model...", flush=True)
            self._load_local()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ----- Image inference (POST /predict) -----

    def predict_annotated(self, image_bytes: bytes) -> tuple[bytes, dict]:
        """Run detection AND return the annotated image with bounding boxes."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        results = self.model(source=image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]
        img_h, img_w = result.orig_shape

        detections = _extract_detections(result.boxes, self._class_map)

        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        annotated_bgr = draw_detections_on_frame(frame_bgr, detections, self._color_map)

        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_bytes = buffer.tobytes()

        result_dict = {
            "detections": detections,
            "num_detections": len(detections),
            "image_width": img_w,
            "image_height": img_h,
            "model_name": self.model_name,
        }

        return annotated_bytes, result_dict

    def process_video_annotated(self, video_bytes: bytes, sample_fps: int = 2) -> tuple[bytes, dict]:
        """Process video AND produce an annotated output video with bounding boxes."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
            tmp_in.write(video_bytes)
            input_path = tmp_in.name

        output_path = input_path.replace(".mp4", "_annotated.mp4")

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / video_fps
            frame_interval = max(1, int(video_fps / sample_fps))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

            frames_data = []
            frame_idx = 0
            total_detections = 0
            class_counts = {}
            current_detections = []
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.model(source=frame_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    current_detections = _extract_detections(results[0].boxes, self._class_map)
                    timestamp = round(frame_idx / video_fps, 2)

                    frames_data.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp,
                        "num_detections": len(current_detections),
                        "detections": current_detections,
                    })

                    total_detections += len(current_detections)
                    for det in current_detections:
                        name = det["class_name"]
                        class_counts[name] = class_counts.get(name, 0) + 1

                annotated_frame = draw_detections_on_frame(frame, current_detections, self._color_map)
                writer.write(annotated_frame)
                frame_idx += 1

            cap.release()
            writer.release()
            processing_time = round(time.time() - start_time, 2)

            with open(output_path, "rb") as f:
                annotated_video_bytes = f.read()

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

        result_dict = {
            "video_info": {
                "width": width, "height": height,
                "fps": round(video_fps, 1), "total_frames": total_frames,
                "duration_sec": round(duration, 2),
            },
            "processing_info": {
                "sample_fps": sample_fps,
                "frames_analyzed": len(frames_data),
                "processing_time_sec": processing_time,
                "model_name": self.model_name,
            },
            "summary": {
                "total_detections": total_detections,
                "class_counts": class_counts,
            },
            "frames": frames_data,
        }

        return annotated_video_bytes, result_dict

    # ----- Single frame inference (WebSocket /ws/detect) -----

    def predict_frame(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a single numpy frame."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        return _extract_detections(results[0].boxes, self._class_map)


# Singleton instance
detector = VehicleDetector()