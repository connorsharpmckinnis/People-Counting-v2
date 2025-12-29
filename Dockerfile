FROM python:3.13-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1



# Install system-level dependencies required by OpenCV and Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# Copy the rest of the project
COPY main.py endpoints.py functions.py job_store.py worker.py /app/
COPY static /app/static

# Create a non-root user for security
RUN useradd --create-home appuser

# Create folders in case they don't already exist
RUN mkdir -p /app/uploads /app/results /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
