#!/bin/bash
source .venv/bin/activate
echo "Starting Demucs ONNX Web Tester (1-second chunk model)..."
echo "Open http://localhost:8000 in your browser."
uvicorn backend.main_chunked:app --port 8000 --reload
