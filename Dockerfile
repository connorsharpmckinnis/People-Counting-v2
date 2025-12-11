FROM python:3.13-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies required by OpenCV and Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create folders in case they don’t already exist
RUN mkdir -p /app/uploads /app/results

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
