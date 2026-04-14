"""
conftest.py — pytest global configuration for AutoVision API.
Sets env vars before any app code is imported.
"""

import os
import pytest

# Force test environment — must happen before app imports
os.environ["MODEL_SOURCE"] = "local"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["TESTING"] = "1"
os.environ["YOLO_OFFLINE"] = "1"   # prevents ultralytics weight downloads


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (>2s)")
    config.addinivalue_line("markers", "integration: requires running services")
