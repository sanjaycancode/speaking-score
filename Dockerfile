# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app


# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Audio processing dependencies
    ffmpeg \
    # eSpeak NG for phonemizer
    espeak-ng \
    espeak-ng-data \
    # Build tools for Python packages
    gcc \
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]