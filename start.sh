#!/bin/bash
cd "$(dirname "$0")"
source "/Users/apple/Desktop/demo audit/files/venv/bin/activate"
mkdir -p uploads
echo "Starting Kalvium Audit Engine on http://localhost:8002"
echo "Open: http://localhost:8002/static/audit.html"
PYTHONPATH="$(pwd)" uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
