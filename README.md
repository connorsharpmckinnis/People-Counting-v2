# People Counting v2

A robust object detection and counting system built with YOLO11 and SAHI (Slicing Aided Hyper Inference). This tool provides both a web UI and API endpoints for processing images and videos to detect and count objects with tracking capabilities.

## Features

### Processing Modes

1. **Basic Image** - Standard YOLO inference on full images
2. **Sliced Image** - SAHI-powered sliced inference for detecting small objects in large images
3. **Video** - Frame-by-frame object detection with ByteTrack tracking
4. **Sliced Video** - Combines sliced inference with video tracking for maximum accuracy

### Key Capabilities

- 🎯 **Object Detection & Counting** - Detect and count multiple object classes
- 🎬 **Video Tracking** - Track unique objects across video frames using ByteTrack
- 🔪 **Sliced Inference** - Process large images/videos in smaller tiles for better small object detection
- 🌐 **Web Interface** - Clean, user-friendly web UI with dynamic settings
- 🔌 **REST API** - FastAPI-based endpoints for programmatic access
- 📹 **Browser-Compatible Video** - Automatic H.264 conversion for in-browser playback
- ⚙️ **Configurable Parameters** - Adjust model, confidence threshold, and slicing parameters

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video H.264 conversion)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd People-Counting-v2
```

2. Install dependencies using `uv` (recommended) or `pip`:
```bash
# Using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

3. Ensure FFmpeg is installed and in your PATH:
```bash
# Windows (using chocolatey)
choco install ffmpeg

# macOS (using homebrew)
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

## Usage

### Starting the Server

```bash
uv run uvicorn endpoints:app --reload
```

The server will start at `http://127.0.0.1:8000`

### Web Interface

1. Open your browser to `http://127.0.0.1:8000`
2. Select a file (image or video)
3. Choose a processing type
4. Configure settings (model, confidence threshold, slicing parameters)
5. Click "Upload & Process"
6. View results and annotated output

### Processing Type Settings

#### Basic Image / Video
- **Model**: YOLO model file (default: `yolo11n.pt`)
- **Confidence Threshold**: Detection confidence (0.0-1.0, default: 0.35)

#### Sliced Image
Additional settings for SAHI sliced inference:
- **Slice Width/Height**: Size of each slice in pixels (default: 256x256)
- **Overlap Width/Height Ratio**: Overlap between slices as ratio (0.0-1.0, default: 0.2)

#### Sliced Video
Additional settings for sliced video processing:
- **Slice Width/Height**: Size of each slice in pixels (default: 960x960)
- **Overlap Width/Height**: Overlap between slices in pixels (default: 10px)

## API Endpoints

### POST `/basic-count`
Process an image with standard YOLO inference.

**Form Data:**
- `file`: Image file
- `model`: Model path (optional, default: "yolo11n.pt")
- `conf_threshold`: Confidence threshold (optional, default: 0.35)

**Response:**
```json
{
  "counts": {"person": 5, "car": 2},
  "annotated_file": "/results/filename_annotated.png"
}
```

### POST `/sliced-count`
Process an image with SAHI sliced inference.

**Form Data:**
- `file`: Image file
- `model`: Model path (optional)
- `conf_threshold`: Confidence threshold (optional)
- `slice_width`: Slice width in pixels (optional, default: 256)
- `slice_height`: Slice height in pixels (optional, default: 256)
- `overlap_width_ratio`: Overlap ratio (optional, default: 0.2)
- `overlap_height_ratio`: Overlap ratio (optional, default: 0.2)

### POST `/video-count`
Process a video with object tracking.

**Form Data:**
- `file`: Video file
- `model`: Model path (optional)
- `conf_threshold`: Confidence threshold (optional)

**Response:**
```json
{
  "counts": {"person": 12, "car": 5},
  "annotated_file": "/results/filename_annotated_h264.mp4"
}
```

### POST `/sliced-video-count`
Process a video with sliced inference and tracking.

**Form Data:**
- `file`: Video file
- `model`: Model path (optional)
- `conf_threshold`: Confidence threshold (optional)
- `slice_width`: Slice width in pixels (optional, default: 960)
- `slice_height`: Slice height in pixels (optional, default: 960)
- `overlap_width`: Overlap in pixels (optional, default: 10)
- `overlap_height`: Overlap in pixels (optional, default: 10)

### GET `/results/{filename}`
Retrieve processed result files (images/videos).

## Project Structure

```
People-Counting-v2/
├── main.py              # Core processing functions
├── endpoints.py         # FastAPI application and routes
├── static/              # Web UI files
│   ├── index.html       # Main HTML interface
│   ├── script.js        # Frontend JavaScript
│   └── style.css        # Styling
├── uploads/             # Temporary upload storage
├── results/             # Processed output files
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Technical Details

### Video Processing Pipeline

1. **Upload** - Video uploaded via web UI or API
2. **Process** - Frame-by-frame detection with optional slicing
3. **Track** - ByteTrack assigns unique IDs to objects across frames
4. **Annotate** - Bounding boxes and labels drawn on frames
5. **Encode** - Initial mp4v encoding for speed
6. **Convert** - FFmpeg converts to H.264 for browser compatibility
7. **Serve** - Video available for playback or download

### Sliced Inference

SAHI (Slicing Aided Hyper Inference) improves detection of small objects by:
- Dividing large images into smaller overlapping tiles
- Running inference on each tile
- Merging predictions with NMS to remove duplicates
- Particularly effective for aerial imagery, crowd scenes, and distant objects

### Object Tracking

ByteTrack tracking provides:
- Unique ID assignment for each detected object
- Consistent tracking across video frames
- Accurate counting of unique objects (not just detections)
- Handles occlusions and temporary disappearances

## Configuration

### Model Selection

The system supports any YOLO11 model. Common options:
- `yolo11n.pt` - Nano (fastest, least accurate)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (slowest, most accurate)

Place custom models in the project directory and reference by filename.

### Confidence Threshold

Lower values (0.1-0.3) detect more objects but may include false positives.
Higher values (0.5-0.8) are more conservative but may miss objects.
Default of 0.35 provides a good balance for most use cases.

## Troubleshooting

### Video Won't Play in Browser
- Ensure FFmpeg is installed and in PATH
- Check server logs for H.264 conversion errors
- Try downloading the video file directly

### Poor Detection Results
- Try sliced inference for small objects
- Adjust confidence threshold
- Use a larger/more accurate model
- Increase slice overlap for better coverage

### Slow Processing
- Use a smaller model (e.g., yolo11n.pt)
- Reduce slice dimensions
- Process shorter video clips
- Consider GPU acceleration (requires CUDA setup)

## Dependencies

- **ultralytics** - YOLO11 implementation
- **sahi** - Slicing Aided Hyper Inference
- **supervision** - Detection utilities and tracking
- **FastAPI** - Web framework
- **OpenCV** - Image/video processing
- **FFmpeg** - Video encoding

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- YOLO11 by Ultralytics
- SAHI by obss
- ByteTrack tracking algorithm
- Supervision library by Roboflow
