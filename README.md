# Detection Engine v2

A robust, secure object detection and counting system built with YOLO11, SAHI (Slicing Aided Hyper Inference), and FastAPI. This tool provides a modern web UI and secure API endpoints for detecting, tracking, and counting objects in images and videos.

**Latest Docker Image:** `connorsharpmckinnis/detection-engine:0.9`

## ✨ New Features in v2

- **🔐 Basic Authentication**: Secure access protected by an application password.
- **🔳 Image Zone Counting**: Draw polygons on images to count objects only within specific zones.
- **🚧 Video Line/Polygon Crossing**: Count objects crossing lines or entering defined zones in videos.
- **🎯 Class Filtering**: Detect specific objects (e.g., only "person" or "car") by COCO class ID.
- **🏎️ CUDA Support**: Automatic GPU acceleration if available.
- **🚀 Dockerized**: Ready-to-deploy container.

---

## 🚀 Quick Start (Docker)

The easiest way to run the application is via Docker.

1. **Create a `.env` file** with your desired password:
   ```bash
   APP_PASSWORD=my_secure_password
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     -p 8000:8000 \
     --env-file .env \
     connorsharpmckinnis/detection-engine:0.9
   ```

3. **Access the UI**:
   - Open browser to `http://localhost:8000`
   - Login with Username: (leave blank or use 'user') / Password: `my_secure_password`

---

## 🛠️ Manual Installation

### Prerequisites
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) (Installed and in system PATH for video processing)
- CUDA-capable GPU (Optional, for faster processing)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd People-Counting-v2
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # OR using pip
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```bash
   APP_PASSWORD=admin  # Set your desired password here
   ```

4. **Start the Server**:
   ```bash
   uv run uvicorn main:app --reload
   ```

5. **Access the Application**:
   Go to `http://127.0.0.1:8000` and log in.

---

## 🖥️ Web Interface Features

The web UI provides an intuitive way to upload files and configure detection settings.

### Processing Modes
1. **Basic Image**: Standard full-frame YOLO detection.
2. **Sliced Image**: Divides large images into tiles (SAHI) to detect small objects.
3. **Basic Video**: Frame-by-frame detection with unique object tracking (ByteTrack).
4. **Sliced Video**: Combines slicing + tracking for small objects in video.
5. **Polygon Crossing (Video)**: Interactive tool to draw a line/zone; counts objects crossing it.
6. **Image Zone Counting**: Interactive tool to draw a polygon on an image; counts objects inside.

### Settings
- **Model**: Select from Nano (fastest) to Extra Large (most accurate), or a custom drone model.
- **Confidence**: Adjust detection sensitivity (0.1 - 1.0).
- **Classes**: Filter specific objects (e.g., input `0` for person, `2` for car).

---

## 🔌 API Reference

All API endpoints are protected and require **HTTP Basic Auth**.

### 1. Basic Count
`POST /basic-count`
- **file**: Image file
- **model**: Model name (default: `yolo11n.pt`)
- **classes**: JSON list of class IDs (e.g., `[0, 2]`)

### 2. Sliced Count (SAHI)
`POST /sliced-count`
- **file**: Image file
- **slice_width/height**: Tile size (e.g., 256)
- **overlap_width/height_ratio**: Overlap (e.g., 0.2)
- **classes**: JSON list of class IDs

### 3. Video Tracking
`POST /video-count`
- **file**: Video file
- **model**: Model path
- **classes**: JSON list of class IDs
- Returns H.264 encoded video with tracking IDs.

### 4. Polygon Crossing (Video)
`POST /polygon-cross-count`
- **file**: Video file
- **region_points**: String representation of list of tuples, e.g., `"[(100,100), (100,500)]"` (vertical line).
- **classes**: JSON list of class IDs

### 5. Image Zone Count
`POST /image-zone-count`
- **file**: Image file
- **region_points**: Polygon coordinates, e.g., `"[(50,50), (200,50), (200,200), (50,200)]"`.
- **classes**: JSON list of class IDs

---

## 📁 Project Structure

```
People-Counting-v2/
├── main.py              # Auth & App Entry Point
├── endpoints.py         # API Route definitions
├── functions.py         # Core logic (YOLO, SAHI, Tracking, Zones)
├── static/              # Frontend (HTML, CSS, JS)
├── uploads/             # Temp storage
├── results/             # Processed outputs
├── training.py          # YOLO Training script
└── Dockerfile           # Container configuration
```

## ⚠️ Troubleshooting

- **401 Unauthorized**: Ensure you are sending the correct password in the Basic Auth header or logging in via the browser.
- **Video Playback Fails**: Confirm `ffmpeg` is installed. The app attempts to convert videos to H.264 automatically.
- **CUDA Not Used**: Ensure `torch.cuda.is_available()` is True. The app defaults to CPU if no GPU is found.
