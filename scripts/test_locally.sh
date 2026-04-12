#!/bin/bash
# ==============================================================
# test_locally.sh — Quick manual test for AutoVision API
# ==============================================================
# Usage:
#   1. Start the API:   docker-compose up --build
#   2. In another tab:  bash scripts/test_locally.sh
#
# Or if running without Docker:
#   1. pip install -r requirements.txt
#   2. uvicorn app.main:app --reload
#   3. bash scripts/test_locally.sh
# ==============================================================

BASE_URL="http://localhost:8000"

echo "=== Testing AutoVision API ==="
echo ""

# 1. Health check
echo "1. Health check..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# 2. Download a sample traffic image
echo "2. Downloading sample traffic image..."
curl -s -o /tmp/traffic_test.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Gersthofstrasse.jpg/640px-Gersthofstrasse.jpg"
echo "   Downloaded to /tmp/traffic_test.jpg"
echo ""

# 3. Run prediction
echo "3. Running prediction..."
curl -s -X POST "$BASE_URL/predict" \
  -F "file=@/tmp/traffic_test.jpg" | python3 -m json.tool
echo ""

echo "=== Done ==="
echo ""
echo "You can also visit http://localhost:8000/docs for the interactive Swagger UI"
