# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies for phonemizer/espeak
RUN apt-get update && \
    apt-get install -y espeak-ng libespeak-ng1 libespeak-ng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install fastapi[standard]

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app with FastAPI CLI
CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0", "--port", "8000"]
