FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1



# Install system-level dependencies required by OpenCV and Ultralytics
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 --break-system-packages

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt --break-system-packages

RUN python -m pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git --break-system-packages

# Copy the rest of the project
COPY main.py endpoints.py functions.py job_store.py worker.py /app/
COPY static /app/static

# Create folders and set permissions
RUN mkdir -p /app/uploads /app/results /app/data /app/models

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
