"""
Pydantic models for AutoVision API request/response validation.
"""

from pydantic import BaseModel, Field


# --- Shared ---

class Detection(BaseModel):
    """A single detected object."""

    class_id: int = Field(description="COCO class ID")
    class_name: str = Field(description="Human-readable class name, e.g. 'car'")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    bbox: list[float] = Field(description="Bounding box [x_min, y_min, x_max, y_max] in pixels")


# --- Image prediction ---

class PredictionResponse(BaseModel):
    """Response from POST /predict (single image)."""

    success: bool = True
    num_detections: int = Field(description="Total objects detected")
    detections: list[Detection] = Field(description="List of detected objects")
    image_width: int = Field(description="Original image width in pixels")
    image_height: int = Field(description="Original image height in pixels")
    model_name: str = Field(description="Name of the model used")


# --- Video prediction ---

class FrameDetections(BaseModel):
    """Detections for a single video frame."""

    frame_index: int = Field(description="Frame number in the original video")
    timestamp_sec: float = Field(description="Timestamp in seconds")
    num_detections: int
    detections: list[Detection]


class VideoInfo(BaseModel):
    """Metadata about the uploaded video."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float


class ProcessingInfo(BaseModel):
    """How the video was processed."""

    sample_fps: int = Field(description="Frames analyzed per second of video")
    frames_analyzed: int = Field(description="Total frames that were analyzed")
    processing_time_sec: float = Field(description="Wall-clock processing time")
    model_name: str


class VideoSummary(BaseModel):
    """Aggregated detection counts across all frames."""

    total_detections: int
    class_counts: dict[str, int] = Field(
        description="How many times each class was detected across all frames"
    )


class VideoResponse(BaseModel):
    """Response from POST /predict/video."""

    success: bool = True
    video_info: VideoInfo
    processing_info: ProcessingInfo
    summary: VideoSummary
    frames: list[FrameDetections]


# --- Health ---

class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = "healthy"
    model_loaded: bool
    model_name: str
