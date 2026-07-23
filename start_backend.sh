#!/bin/bash
# Script to run the Columbia Inventory agent toolset API

# Navigate to the script's directory
cd "$(dirname "$0")"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install/Update dependencies
echo "Verifying dependencies..."
pip install -r requirements.txt

# Launch FastAPI app using uvicorn
echo "Starting uvicorn server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
