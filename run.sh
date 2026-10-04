#!/bin/bash
echo "Starting LLM Gateway..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Start the app
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
