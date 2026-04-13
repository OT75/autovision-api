# ==============================================================
# AutoVision API — Multi-stage Dockerfile
# ==============================================================
# Why multi-stage?
# - Build stage: installs all dependencies (big, slow, has compilers)
# - Runtime stage: copies only what's needed (smaller, faster, safer)
# You already know this from your Docker week — this is real-world usage.
# ==============================================================

# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed to compile some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Only the runtime libraries (no compilers, no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ app/

# Set a writable YOLO config dir so the non-root user can write settings at runtime
ENV YOLO_CONFIG_DIR=/app/.ultralytics

# Download yolov8n.pt (fallback model) into YOLO_CONFIG_DIR while still root
RUN mkdir -p /app/.ultralytics && \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Non-root user for security — give them ownership of /app (including .ultralytics)
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port
EXPOSE 8000

# Health check — Docker and orchestrators use this
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
