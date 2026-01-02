import os
import uuid
from fastapi import File, UploadFile, Form, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Tuple
from pathlib import Path
from job_store import create_job, get_job

# Import only estimation functions here; actual processing is in worker.py
from functions import estimate_image_complexity, estimate_video_complexity

router = APIRouter()

TOKENS = {}

# Folder to store uploaded files and results
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def save_upload_file(upload_file: UploadFile, dest: Path) -> Path:
    """Save uploaded file to destination path"""
    with open(dest, "wb") as f:
        f.write(upload_file.file.read())
    return dest

def is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]



@router.post("/basic-count")
async def api_basic_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_image(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts image files.")

    # Parse classes
    class_list = None
    if classes:
        try:
            import json
            class_list = json.loads(classes)
        except:
            print(f"Failed to parse classes: {classes}")

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "classes": class_list
    }
    
    job_id = create_job("basic", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.post("/sliced-count")
async def api_sliced_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    slice_width: Optional[int] = Form(256),
    slice_height: Optional[int] = Form(256),
    overlap_width_ratio: Optional[float] = Form(0.2),
    overlap_height_ratio: Optional[float] = Form(0.2),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_image(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts image files.")

    # Parse classes
    class_list = None
    if classes:
        try:
            import json
            class_list = json.loads(classes)
        except:
            print(f"Failed to parse classes: {classes}")

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "slice_width": slice_width,
        "slice_height": slice_height,
        "overlap_width_ratio": overlap_width_ratio,
        "overlap_height_ratio": overlap_height_ratio,
        "classes": class_list
    }

    job_id = create_job("sliced", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.post("/video-count")
async def api_video_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_video(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts video files.")

    # Parse classes
    class_list = None
    if classes:
        try:
            import json
            class_list = json.loads(classes)
        except:
            print(f"Failed to parse classes: {classes}")

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "classes": class_list
    }

    job_id = create_job("video", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})

@router.post("/sliced-video-count")
async def api_sliced_video_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    slice_width: Optional[int] = Form(960),
    slice_height: Optional[int] = Form(960),
    overlap_width: Optional[int] = Form(10),
    overlap_height: Optional[int] = Form(10),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_video(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts video files.")

    # Parse classes
    class_list = None
    if classes:
        try:
            import json
            class_list = json.loads(classes)
        except:
            print(f"Failed to parse classes: {classes}")

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "slice_wh": (slice_width, slice_height),
        "overlap_wh": (overlap_width, overlap_height),
        "classes": class_list
    }

    job_id = create_job("sliced_video", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})

from pydantic import Json
import json

@router.post("/polygon-cross-count")
async def api_polygon_cross_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    region_points: Optional[str] = Form("[(300, 100), (300, 1200)]"),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_video(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts video files.")

    # Parse region points string - supports both list (single region) and dict (multiple regions)
    try:
        import ast
        points = ast.literal_eval(region_points)
        # Validate format: either a list of tuples (single region) or dict of lists (multi-region)
        if isinstance(points, dict):
            # Multi-region format: {"region-01": [(x,y), ...], "region-02": [...]}
            for key, val in points.items():
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"Region {key} must be a list of points")
        elif not isinstance(points, (list, tuple)):
            raise ValueError("region_points must be a list or dict")
    except Exception as e:
        points = [(0, 0), (100, 100)] 
        print(f"Failed to parse region points: {region_points}, error: {e}, using default.")

    # Parse classes
    class_list = None
    if classes:
        try:
            import json
            class_list = json.loads(classes)
        except:
            print(f"Failed to parse classes: {classes}")

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "region_points": points,
        "classes": class_list
    }

    job_id = create_job("polygon_cross_count", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})






@router.post("/image-zone-count")
async def api_image_zone_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    region_points: Optional[str] = Form("[(50, 50), (250, 50), (250, 250), (50, 250)]"),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_image(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts image files.")

    # Parse region points string - supports both list (single region) and dict (multiple regions)
    try:
        import ast
        points = ast.literal_eval(region_points)
        # Validate format: either a list of tuples (single region) or dict of lists (multi-region)
        if isinstance(points, dict):
            # Multi-region format: {"region-01": [(x,y), ...], "region-02": [...]}
            for key, val in points.items():
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"Region {key} must be a list of points")
        elif not isinstance(points, (list, tuple)):
            raise ValueError("region_points must be a list or dict")
    except Exception as e:
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        print(f"Failed to parse region points: {region_points}, error: {e}, using default.")

    # Parse classes
    class_list = None
    if classes:
        try:
            import json
            class_list = json.loads(classes)
        except:
            print(f"Failed to parse classes: {classes}")

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "region_points": points,
        "classes": class_list
    }

    job_id = create_job("image_zone_count", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})

@router.get("/secure-results/{filename}")
async def secure_image(filename: str):
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(file_path)
    
@router.get("/get-result/{filename}")
async def get_result_file(filename: str):
    """Serve result files directly with proper headers"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4" if filename.endswith(".mp4") else "image/png",
        filename=filename
    )


@router.post("/image-custom-count")
async def api_image_custom_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolov8s-world.pt"),
    conf_threshold: Optional[float] = Form(0.20),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_image(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts image files.")

    class_list = []
    if classes:
        class_list = [c.strip() for c in classes.split(",") if c.strip()]
    if not class_list:
        class_list = ["person"]

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "classes": class_list
    }

    job_id = create_job("image_custom", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.post("/video-custom-count")
async def api_video_custom_count(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolov8s-world.pt"),
    conf_threshold: Optional[float] = Form(0.20),
    classes: Optional[str] = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    if not is_video(str(filename)):
        os.remove(filename)
        raise HTTPException(status_code=400, detail="This endpoint only accepts video files.")

    class_list = []
    if classes:
        class_list = [c.strip() for c in classes.split(",") if c.strip()]
    if not class_list:
        class_list = ["person"]

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "classes": class_list
    }

    job_id = create_job("video_custom", str(filename), config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.post("/stream-count")
async def api_stream_count(
    request: Request,
    youtube_url: str = Form(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    duration: Optional[int] = Form(30), # Seconds to observe
    frame_skip: Optional[int] = Form(5),
    classes: Optional[str] = Form(None)
):
    # Parse classes
    class_list = None
    if classes:
        try:
            import json
            class_list = json.loads(classes)
        except:
            print(f"Failed to parse classes: {classes}")

    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "duration": duration,
        "frame_skip": frame_skip,
        "classes": class_list,
        "show_window": False # Always false for API usage
    }

    # Create job with the URL as the filename/source
    job_id = create_job("stream", youtube_url, config)
    request.app.state.queue.put(job_id)

    return JSONResponse({"job_id": job_id, "status": "queued"})


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    response = {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "error": job["error"]
    }
    
    if job["result"]:
        import json
        response["result"] = json.loads(job["result"])
        
    return response

@router.post("/estimate-count")
async def api_estimate_count(
    file: UploadFile = File(...),
    slice_width: Optional[int] = Form(None),
    slice_height: Optional[int] = Form(None),
    overlap_width_ratio: Optional[float] = Form(None),
    overlap_height_ratio: Optional[float] = Form(None),
    overlap_width: Optional[int] = Form(None),
    overlap_height: Optional[int] = Form(None),
):
    token = uuid.uuid4().hex
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower()
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    # config construction
    config = {}
    if slice_width: config["slice_width"] = slice_width
    if slice_height: config["slice_height"] = slice_height
    if overlap_width_ratio: config["overlap_width_ratio"] = overlap_width_ratio
    if overlap_height_ratio: config["overlap_height_ratio"] = overlap_height_ratio
    if overlap_width: config["overlap_width"] = overlap_width
    if overlap_height: config["overlap_height"] = overlap_height

    # Logic to pick image vs video
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        result = estimate_image_complexity(str(filename), config)
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        result = estimate_video_complexity(str(filename), config)
    else:
        result = {"error": f"Unsupported file type: {ext}"}

    # Clean up temp file immediately? 
    # For now, yes, as we just want an estimate.
    try:
        os.remove(filename)
    except:
        pass

    return JSONResponse(result)