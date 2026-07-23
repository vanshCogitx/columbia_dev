FROM python:3.12-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (including pre-built data/inventory.db)
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI app with Uvicorn, using the PORT environment variable if provided
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
