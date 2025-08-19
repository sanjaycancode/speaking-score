# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies for phonemizer (espeak-ng) and whisper (ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    espeak-ng libespeak-ng1 libespeak-ng-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app with FastAPI CLI
CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0", "--port", "8000"]