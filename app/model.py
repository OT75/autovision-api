"""
YOLO model wrapper for AutoVision API.

This module handles loading the model and running inference.
It's separated from the API routes so you can:
- Swap models easily (Phase 2: load from MLflow registry)
- Test inference without starting the server
- Reuse the logic in training scripts

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
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

# Vehicle-related COCO class IDs we care about for traffic detection
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

# Colors for drawing bounding boxes (BGR for OpenCV)
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


def _extract_detections(boxes) -> list[dict]:
    """Extract and filter detections from YOLO result boxes.

    Shared logic used by image, video, and webcam inference.
    """
    detections = []
    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        if class_id not in VEHICLE_CLASSES:
            continue
        detections.append({
            "class_id": class_id,
            "class_name": VEHICLE_CLASSES[class_id],
            "confidence": round(float(boxes.conf[i].item()), 4),
            "bbox": [round(c, 2) for c in boxes.xyxy[i].tolist()],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_detections_on_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame.

    Used by both video processing and webcam streaming
    so the visual style is consistent everywhere.
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        class_id = det["class_id"]
        label = f"{det['class_name']} {det['confidence']:.0%}"
        color = CLASS_COLORS.get(class_id, (200, 200, 200))

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 4, y1),
            color, -1,
        )
        # Label text
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

    def load(self):
        """Load the YOLO model. Called once at startup."""
        print(f"Loading model: {self.model_name}")
        self.model = YOLO(self.model_name)
        print(f"Model loaded successfully: {self.model_name}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ----- Image inference (POST /predict) -----

    def predict_annotated(self, image_bytes: bytes) -> tuple[bytes, dict]:
        """
        Run detection AND return the annotated image with bounding boxes.

        Returns:
            (annotated_jpeg_bytes, result_dict)

        Used by POST /predict/image — gives you back a JPEG you can
        actually look at to verify predictions are correct.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        results = self.model(source=image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]
        img_h, img_w = result.orig_shape

        detections = _extract_detections(result.boxes)

        # Draw boxes on the image (convert RGB -> BGR for OpenCV, then back)
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        annotated_bgr = draw_detections_on_frame(frame_bgr, detections)

        # Encode to JPEG
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
        """
        Process video AND produce an annotated output video with bounding boxes.

        Returns:
            (annotated_mp4_bytes, result_dict)

        The output video has bounding boxes drawn on every sampled frame.
        Non-sampled frames are passed through unchanged (so the video
        plays smoothly, but boxes only update at sample_fps rate).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
            tmp_in.write(video_bytes)
            input_path = tmp_in.name

        # Output path for annotated video
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

            # Use mp4v codec — widely compatible
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

            frames_data = []
            frame_idx = 0
            total_detections = 0
            class_counts = {}
            current_detections = []  # persist detections between sampled frames
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.model(source=frame_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    current_detections = _extract_detections(results[0].boxes)
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

                # Draw current detections on every frame (smooth playback)
                annotated_frame = draw_detections_on_frame(frame, current_detections)
                writer.write(annotated_frame)

                frame_idx += 1

            cap.release()
            writer.release()
            processing_time = round(time.time() - start_time, 2)

            # Read the output video
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
        """
        Run detection on a single numpy frame.
        Used by WebSocket webcam streaming — needs to be fast.
        Returns just the detections list (no metadata overhead).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        return _extract_detections(results[0].boxes)


# Singleton instance
detector = VehicleDetector()
