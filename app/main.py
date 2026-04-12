"""
AutoVision API — Traffic & Vehicle Detection Service

Endpoints:
- GET  /health          → service health check
- POST /predict         → upload image, get annotated JPEG
- POST /predict/video   → upload video, get annotated MP4
- GET  /stream          → webcam live detection page (from stream router)
- WS   /ws/detect       → WebSocket for real-time detection (from stream router)
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from app.model import detector
from app.schemas import HealthResponse
from app.stream import router as stream_router


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load()
    yield


# --- App ---
app = FastAPI(
    title="AutoVision API",
    description="Traffic & vehicle detection using YOLOv8. Supports images, video files, and real-time webcam streaming.",
    version="0.2.0",
    lifespan=lifespan,
)

# Mount the streaming router (WebSocket + webcam page)
app.include_router(stream_router)


# --- Config ---
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime", "video/webm"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024     # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024    # 100 MB


# ================================================================
# 1. HEALTH
# ================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if the service is running and the model is loaded."""
    return HealthResponse(
        status="healthy" if detector.is_loaded else "unhealthy",
        model_loaded=detector.is_loaded,
        model_name=detector.model_name,
    )


# ================================================================
# 2. IMAGE PREDICTION — returns annotated JPEG
# ================================================================

@app.post("/predict", summary="Upload image, get annotated JPEG with bounding boxes")
async def predict_image(
    file: UploadFile = File(description="Image file (JPEG, PNG, or WebP)"),
):
    """
    Upload an image and get it back with bounding boxes drawn on it.
    Detection JSON is in the X-Detections response header.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=422, detail=f"File too large. Max: {MAX_IMAGE_SIZE // (1024*1024)} MB.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty file uploaded.")

    try:
        annotated_bytes, result = detector.predict_annotated(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    return Response(
        content=annotated_bytes,
        media_type="image/jpeg",
        headers={
            "X-Detections": json.dumps(result),
            "Content-Disposition": "inline; filename=annotated.jpg",
        },
    )


# ================================================================
# 3. VIDEO PREDICTION — returns annotated MP4
# ================================================================

@app.post("/predict/video", summary="Upload video, get annotated MP4 with bounding boxes")
async def predict_video(
    file: UploadFile = File(description="Video file (MP4, AVI, WebM, MOV)"),
    sample_fps: int = Query(
        default=2, ge=1, le=30,
        description="Frames to analyze per second of video.",
    ),
):
    """
    Upload a video and get it back with bounding boxes on every frame.
    Summary JSON is in the X-Summary response header.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )

    video_bytes = await file.read()

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=422, detail=f"File too large. Max: {MAX_VIDEO_SIZE // (1024*1024)} MB.")

    if len(video_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty file uploaded.")

    try:
        annotated_video_bytes, result = detector.process_video_annotated(
            video_bytes, sample_fps=sample_fps
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

    summary_header = json.dumps({
        "video_info": result["video_info"],
        "processing_info": result["processing_info"],
        "summary": result["summary"],
    })

    return Response(
        content=annotated_video_bytes,
        media_type="video/mp4",
        headers={
            "X-Summary": summary_header,
            "Content-Disposition": "attachment; filename=annotated_output.mp4",
        },
    )


# --- Global exception handler ---

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Internal server error"},
    )
