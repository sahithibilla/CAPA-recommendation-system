#!/usr/bin/env bash
# Run the CAPA Recommendation System backend
set -e

cd "$(dirname "$0")/backend"

echo "==================================================="
echo "  CAPA AI Recommendation System - Backend"
echo "==================================================="
echo ""

# Check Python version
python3 --version

# Install dependencies if needed
echo "[1/2] Checking dependencies..."
pip install -r requirements.txt --quiet

echo "[2/2] Starting FastAPI server..."
echo ""
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo ""
echo "  Open frontend/index.html in your browser."
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
