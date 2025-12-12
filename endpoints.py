import os
import uuid
from fastapi import File, UploadFile, Form, APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from pathlib import Path

# Import your existing counting functions
from functions import basic_count, sliced_count, video_count, sliced_video_count

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


@router.post("/basic-count")
async def api_basic_count(
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35)
):

    token = uuid.uuid4().hex

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix  # ".jpg" or ".png" or ".mp4"
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    counts, annotated_path = basic_count(str(filename), config={"model": model, "conf_threshold": conf_threshold})

    # Move annotated image to RESULTS_DIR
    annotated_file = RESULTS_DIR / f"{token}{ext}"
    os.replace(annotated_path, annotated_file)

    TOKENS[token] = annotated_file

    print("RESPONSE DEBUG:", {
        "counts": counts,
        "url": f"/secure-results/{token}{ext}",
        "file_type": "image"
    })

    return JSONResponse({"counts": counts, "annotated_file": f"/secure-results/{token}{ext}", "file_type": "image"})


@router.post("/sliced-count")
async def api_sliced_count(
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    slice_width: Optional[int] = Form(256),
    slice_height: Optional[int] = Form(256),
    overlap_width_ratio: Optional[float] = Form(0.2),
    overlap_height_ratio: Optional[float] = Form(0.2)
):

    token = uuid.uuid4().hex

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix  # ".jpg" or ".png" or ".mp4"
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    # Build config with slice parameters
    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "slice_width": slice_width,
        "slice_height": slice_height,
        "overlap_width_ratio": overlap_width_ratio,
        "overlap_height_ratio": overlap_height_ratio
    }

    counts, annotated_path = sliced_count(str(filename), config=config)

    annotated_file = RESULTS_DIR / f"{token}{ext}"
    os.replace(annotated_path, annotated_file)

    TOKENS[token] = annotated_file

    return JSONResponse({"counts": counts, "annotated_file": f"/secure-results/{token}{ext}", "file_type": "image"})


@router.post("/video-count")
async def api_video_count(
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35)
):

    token = uuid.uuid4().hex

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix  # ".jpg" or ".png" or ".mp4"
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    counts, annotated_path = video_count(str(filename), config={"model": model, "conf_threshold": conf_threshold})

    annotated_file = RESULTS_DIR / f"{token}{ext}"
    
    # Check if the source file exists before moving
    if not os.path.exists(annotated_path):
        return JSONResponse({"error": f"Video processing failed - output file not created: {annotated_path}"}, status_code=500)
    
    os.replace(annotated_path, annotated_file)
    
    # Verify the file was moved successfully
    if not annotated_file.exists():
        return JSONResponse({"error": f"Failed to move annotated file to results directory"}, status_code=500)

    TOKENS[token] = annotated_file

    return JSONResponse({"counts": counts, "annotated_file": f"/secure-results/{token}{ext}", "file_type": "video"})

@router.post("/sliced-video-count")
async def api_sliced_video_count(
    file: UploadFile = File(...),
    model: Optional[str] = Form("yolo11n.pt"),
    conf_threshold: Optional[float] = Form(0.35),
    slice_width: Optional[int] = Form(960),
    slice_height: Optional[int] = Form(960),
    overlap_width: Optional[int] = Form(10),
    overlap_height: Optional[int] = Form(10)
):

    token = uuid.uuid4().hex

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix  # ".jpg" or ".png" or ".mp4"
    filename = UPLOAD_DIR / f"{file_id}{ext}"
    save_upload_file(file, filename)

    # Build config with slice parameters
    config = {
        "model": model,
        "conf_threshold": conf_threshold,
        "slice_wh": (slice_width, slice_height),
        "overlap_wh": (overlap_width, overlap_height)
    }

    counts, annotated_path = sliced_video_count(str(filename), config=config)

    annotated_file = RESULTS_DIR / f"{token}{ext}"
    os.replace(annotated_path, annotated_file)

    TOKENS[token] = annotated_file

    return JSONResponse({"counts": counts, "annotated_file": f"/secure-results/{token}{ext}", "file_type": "video"})

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