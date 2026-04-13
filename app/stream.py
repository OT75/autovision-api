"""
Stream router — real-time webcam detection via WebSocket.

Separated from main.py because:
- The HTML template alone is ~250 lines
- The WebSocket logic is independent from the REST endpoints
- You can disable streaming entirely by just not including this router

This is a FastAPI APIRouter — it gets mounted into the main app
with app.include_router(). Same pattern companies use to organize
large APIs into logical modules.
"""

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.model import detector, draw_detections_on_frame

router = APIRouter()

# Load the HTML template once at import time
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "stream.html"
_STREAM_HTML = _TEMPLATE_PATH.read_text()


@router.get("/stream", response_class=HTMLResponse)
async def stream_page():
    """
    Serves the webcam live detection page.
    Open http://localhost:8000/stream in your browser.
    """
    return HTMLResponse(content=_STREAM_HTML)


@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket for real-time frame-by-frame detection.

    Protocol:
    1. Client sends a frame as base64-encoded JPEG
    2. Server runs YOLO, draws bounding boxes
    3. Server sends back annotated frame + detection JSON
    4. Repeat

    The client controls the frame rate by waiting for each response
    before sending the next frame — no queue buildup.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            frame_b64 = msg.get("frame", "")

            # Decode base64 -> numpy array
            frame_bytes = base64.b64decode(frame_b64)
            frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            # Run detection
            start = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.predict_frame(frame_rgb)
            inference_ms = round((time.time() - start) * 1000, 1)

            # Draw boxes and encode back to JPEG
            annotated = draw_detections_on_frame(frame, detections, detector._color_map)
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_json({
                "frame": annotated_b64,
                "detections": detections,
                "num_detections": len(detections),
                "inference_ms": inference_ms,
            })

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except RuntimeError:
            pass
