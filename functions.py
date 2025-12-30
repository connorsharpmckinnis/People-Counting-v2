
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from collections import Counter
from ultralytics import YOLO, solutions, YOLOWorld, YOLOE
from ultralytics.trackers.byte_tracker import BYTETracker
from collections import defaultdict
import cv2
import numpy as np
import supervision as sv
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TypedDict, Optional, List
import torch

# Directory to store model weight files (.pt)
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(model_name: str) -> str:
    """
    Always resolve model paths into MODEL_DIR.

    - Bare names (e.g. 'yolo11n.pt') → /app/models/yolo11n.pt
    - Relative paths → /app/models/<basename>
    - Absolute paths → left untouched
    """
    if not model_name:
        return model_name

    # Absolute paths are respected (power users / tests)
    if os.path.isabs(model_name):
        return model_name

    # Strip any directory parts and force into MODEL_DIR
    model_filename = os.path.basename(model_name)
    return os.path.join(MODEL_DIR, model_filename)


class Config(TypedDict):
    model: str
    """Path to model weights"""
    classes: Optional[List[str]]
    """List of classes to track"""
    tracker: Optional[str]
    """Tracker to use"""
    conf_threshold: float
    """Confidence threshold for object detection"""
    slice_wh: Optional[tuple]
    """Slice width and height"""
    overlap_wh: Optional[tuple]
    """Overlap width and height"""
    slice_height: Optional[int]
    """Slice height"""
    slice_width: Optional[int]
    """Slice width"""
    overlap_height_ratio: Optional[float]
    """Overlap height ratio"""
    overlap_width_ratio: Optional[float]
    """Overlap width ratio"""
    region_points: Optional[List[tuple]]
    """Region points"""
    output_path: Optional[str]
    """Output path"""
    input_path: Optional[str]
    """Input path"""

def calculate_slices(image_wh: tuple, slice_wh: tuple, overlap_wh: tuple) -> int:
    """
    Calculate the number of slices for a given image resolution and slice configuration.
    Mimics typical tiling logic (e.g. SAHI).
    """
    import math
    img_w, img_h = image_wh
    slice_w, slice_h = slice_wh
    overlap_w, overlap_h = overlap_wh

    # Basic validation
    if slice_w <= 0 or slice_h <= 0:
        return 1
    
    # If image is smaller than slice, it's just 1 slice (typically padded)
    if img_w <= slice_w and img_h <= slice_h:
        return 1

    stride_w = slice_w - overlap_w
    stride_h = slice_h - overlap_h

    # Prevent infinite loop or div by zero if overlap >= slice
    if stride_w <= 0: stride_w = slice_w
    if stride_h <= 0: stride_h = slice_h

    # Calculate steps in x and y
    # We want to cover the whole image. 
    # Logic: how many strides fit? +1 for the first slice.
    # The last slice might overlap more than overlap_wh to align with edge, but it's still a slice.
    
    n_w = math.ceil((img_w - slice_w) / stride_w) + 1 if img_w > slice_w else 1
    n_h = math.ceil((img_h - slice_h) / stride_h) + 1 if img_h > slice_h else 1

    return n_w * n_h

def estimate_image_complexity(image_path: str, config: dict) -> dict:
    """
    Estimate processing steps for an image.
    """
    try:
        # Just read metadata if possible, but cv2.imread is reliable. 
        # For huge images, this might be slowish but unavoidable to get accurate resolution if not trusted.
        # Alternatively use PIL for faster header read.
        image = cv2.imread(image_path)
        if image is None: 
            return {"error": "Could not read image"}
        
        h, w = image.shape[:2]
        
        slice_wh = config.get("slice_wh")
        overlap_wh = config.get("overlap_wh")
         
        # Ensure they are tuples of ints if they came back from JSON as lists
        if isinstance(slice_wh, (list, tuple)):
            slice_wh = tuple(int(x) for x in slice_wh)
        if isinstance(overlap_wh, (list, tuple)):
            overlap_wh = tuple(int(x) for x in overlap_wh)

        # Check if slice params exist separately in config (endpoints might pass them differently)
        if not slice_wh:
            sw = config.get("slice_width")
            sh = config.get("slice_height")
            if sw and sh:
                slice_wh = (int(sw), int(sh))
        
        if not overlap_wh:
            # Try ratio first
            ow_r = config.get("overlap_width_ratio")
            oh_r = config.get("overlap_height_ratio")
            if ow_r and oh_r and slice_wh:
                overlap_wh = (int(slice_wh[0] * ow_r), int(slice_wh[1] * oh_r))
        
        slices = 1
        if slice_wh and overlap_wh:
             slices = calculate_slices((w, h), slice_wh, overlap_wh)
        
        return {
            "file_type": "image",
            "resolution": (w, h),
            "total_frames": 1,
            "slices_per_frame": slices,
            "total_inference_steps": slices
        }
    except Exception as e:
        return {"error": str(e)}

def estimate_video_complexity(video_path: str, config: dict) -> dict:
    """
    Estimate processing steps for a video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             return {"error": "Could not read video"}
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        duration = frame_count / fps if fps else 0
        
        slice_wh = config.get("slice_wh")
        overlap_wh = config.get("overlap_wh")
         
        # Ensure they are tuples of ints if they came back from JSON as lists
        if isinstance(slice_wh, (list, tuple)):
            slice_wh = tuple(int(x) for x in slice_wh)
        if isinstance(overlap_wh, (list, tuple)):
            overlap_wh = tuple(int(x) for x in overlap_wh)

        # Check if slice params exist separately
        if not slice_wh:
            sw = config.get("slice_width")
            sh = config.get("slice_height")
            if sw and sh:
                slice_wh = (int(sw), int(sh))
        
        if not overlap_wh and slice_wh:
            # Maybe passed as loose keys
            ow = config.get("overlap_width")
            oh = config.get("overlap_height")
            if ow is not None and oh is not None:
                overlap_wh = (int(ow), int(oh))


        # If we have slice config, calculate. Otherwise 1 slice (whole frame).
        slices_per_frame = 1
        if slice_wh and overlap_wh: # Only if explicitly slicing
            slices_per_frame = calculate_slices((w, h), slice_wh, overlap_wh)
        elif config.get("slice_width"): # Maybe passed as loose keys
            sw = int(config.get("slice_width"))
            sh = int(config.get("slice_height"))
            ow = int(config.get("overlap_width", 0))
            oh = int(config.get("overlap_height", 0))
            slices_per_frame = calculate_slices((w, h), (sw, sh), (ow, oh))

        total_steps = frame_count * slices_per_frame
        
        return {
            "file_type": "video",
            "resolution": (w, h),
            "total_frames": frame_count,
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "slices_per_frame": slices_per_frame,
            "total_inference_steps": total_steps
        }
    except Exception as e:
        return {"error": str(e)}

def convert_to_h264(input_path: str, output_path: str) -> bool:
    """Convert video to H.264 format for browser compatibility using ffmpeg"""
    try:
        # Use ffmpeg to convert to H.264
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',
            '-y',  # Overwrite output file
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        return False

def save_image(source_filepath: str, content_to_save) -> str:
    base, ext = os.path.splitext(source_filepath)
    out_path = f"{base}_annotated{ext}"
    
    cv2.imwrite(out_path, content_to_save)

    return out_path 

def annotate_image(image_path: str, detections: sv.Detections, labels: list):
    image = cv2.imread(image_path)
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=2,
        text_padding=2
    )
    annotated = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )
    return annotated








def basic_count(image_path: str, config: dict) -> tuple[dict, str]:


    model_name = config.get("model", "yolo11n.pt")
    model_path = get_model_path(model_name)
    conf_threshold = config.get("conf_threshold", 0.35)

    model = YOLO(model_path)

    # Perform inference
    results = model(image_path, conf=conf_threshold, classes=config.get("classes"))

    # Convert YOLO results → Supervision Detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Build labels like "person", "car", etc.
    labels = [model.names[int(cls_id)] for cls_id in detections.class_id]
    counts = Counter(labels)

    # Annotate
    annotated = annotate_image(image_path, detections, labels)

    # Save annotated image
    out_path = save_image(image_path, annotated)

    return dict(counts), out_path

def sliced_count(image_path: str, config: dict) -> tuple[dict, str]:
    model_name = config.get("model", "yolo11n.pt")
    model_path = get_model_path(model_name)
    conf_threshold = config.get("conf_threshold", 0.35)
    slice_height = config.get("slice_height", 256)
    slice_width = config.get("slice_width", 256)
    overlap_height_ratio = config.get("overlap_height_ratio", 0.2)
    overlap_width_ratio = config.get("overlap_width_ratio", 0.2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device,
    )

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    # Optional class filtering
    classes_to_track = config.get("classes")
    if classes_to_track is not None:
        result.object_prediction_list = [
            pred for pred in result.object_prediction_list
            if pred.category.id in classes_to_track
        ]

    # ---- Convert SAHI predictions → Supervision Detections ----
    xyxy = []
    class_ids = []
    scores = []

    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        xyxy.append([x1, y1, x2, y2])
        class_ids.append(pred.category.id)
        scores.append(pred.score.value)

    if len(xyxy) == 0:
        return {}, image_path

    detections = sv.Detections(
        xyxy=np.array(xyxy),
        class_id=np.array(class_ids),
        confidence=np.array(scores)
    )

    # ---- Build labels ----
    category_names = detection_model.model.names
    labels = [
        f"{category_names[class_id]}"
        for class_id in detections.class_id
    ]

    # ---- Annotate with Supervision ----
    annotated = annotate_image(image_path, detections, labels)
    
    out_path =save_image(image_path, annotated)

    counts = Counter(labels)

    return dict(counts), out_path

def video_count(video_path: str, config: dict) -> tuple[dict, str]:
    model_name = config.get("model", "yolo11n.pt")
    model_path = get_model_path(model_name)
    conf_threshold = config.get("conf_threshold", 0.35)

    model = YOLO(model_path)

    # Prepare output path
    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_annotated.mp4"  # Force .mp4 extension

    # Grab video metadata for writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v codec (most reliable on Windows)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.8,
        text_thickness=2,
        text_color=sv.Color.WHITE
    )

    # YOLO with tracking
    results_stream = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        stream=True,
        conf=conf_threshold,
        classes=config.get("classes")
    )

    seen_ids = {}
    class_names = model.names

    for frame_results in results_stream:
        frame = frame_results.orig_img.copy()

        # Convert YOLO result → Supervision Detections
        det = sv.Detections.from_ultralytics(frame_results)

        if det.tracker_id is None or len(det) == 0:
            writer.write(frame)
            continue

        # Only keep detections with assigned track IDs
        det = det[det.tracker_id != None]

        # Generate labels like "person #12"
        labels = []
        for cls_id, track_id in zip(det.class_id, det.tracker_id):
            cls_name = class_names[cls_id]
            labels.append(f"{cls_name} #{track_id}")

            # Track unique IDs per class
            seen_ids.setdefault(cls_name, set()).add(int(track_id))

        # Annotate
        annotated = box_annotator.annotate(scene=frame, detections=det)
        annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)

        writer.write(annotated)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    # Small delay to ensure file is fully written and unlocked on Windows
    time.sleep(0.5)

    # Convert to H.264 for browser compatibility
    base, ext = os.path.splitext(out_path)
    h264_path = f"{base}_h264.mp4"
    
    if convert_to_h264(out_path, h264_path):
        # Conversion successful, remove the mp4v version and use H.264
        os.remove(out_path)
        final_path = h264_path
    else:
        # Conversion failed, use the mp4v version (user can download it)
        print("Warning: H.264 conversion failed, using mp4v format")
        final_path = out_path

    count_dict = {cls: len(ids) for cls, ids in seen_ids.items()}
    return count_dict, final_path

def sliced_video_count(
    video_path: str, 
    config: dict
) -> tuple[dict, str]:
    model_name = config.get("model", "yolo11n.pt")
    model_path = get_model_path(model_name)
    base, ext = os.path.splitext(video_path)
    save_path = f"{base}_annotated.mp4"  # Force .mp4 extension
    slice_wh = config.get("slice_wh") or (960, 960)
    # Ensure slice_wh is a tuple of integers
    if isinstance(slice_wh, (list, tuple)):
         slice_wh = tuple(int(x) for x in slice_wh)
    
    overlap_wh = config.get("overlap_wh") or (10, 10)
    if isinstance(overlap_wh, (list, tuple)):
        overlap_wh = tuple(int(x) for x in overlap_wh)

    conf_threshold = config.get("conf_threshold", 0.50)
    
    
    model = YOLO(model_path)
    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    
    unique_ids = {}

    def slice_callback(image_slice: np.ndarray) -> sv.Detections:
        results = model(image_slice, conf=conf_threshold, classes=config.get("classes"))[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    slicer = sv.InferenceSlicer(
        callback=slice_callback,
        slice_wh=slice_wh,
        overlap_wh=overlap_wh,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
         raise RuntimeError("Error opening video file")
    
    # Prepare video writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Use mp4v codec (most reliable on Windows)
    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )


    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = slicer(frame)

        detections = tracker.update_with_detections(detections)

        for idx, track_id in enumerate(detections.tracker_id):
            class_id = int(detections.class_id[idx])
            class_name = model.names[class_id]
            
            if class_name not in unique_ids:
                unique_ids[class_name] = set()
            
            unique_ids[class_name].add(track_id)
    
        # ---- ANNOTATION ----
        labels = [
            f"{model.names[int(c)]} #{tid}"
            for c, tid in zip(detections.class_id, detections.tracker_id)
        ]

        annotated = box_annotator.annotate(frame.copy(), detections=detections)
        annotated = label_annotator.annotate(
            annotated,
            detections=detections,
            labels=labels
        )

        

        # save to file
        writer.write(annotated)

    cap.release()
    writer.release()
    
    # Small delay to ensure file is fully written and unlocked on Windows
    time.sleep(0.5)
    
    # Convert to H.264 for browser compatibility
    base, ext = os.path.splitext(save_path)
    h264_path = f"{base}_h264.mp4"
    
    if convert_to_h264(save_path, h264_path):
        # Conversion successful, remove the mp4v version and use H.264
        os.remove(save_path)
        final_path = h264_path
    else:
        # Conversion failed, use the mp4v version (user can download it)
        print("Warning: H.264 conversion failed, using mp4v format")
        final_path = save_path

    final_counts = {cls: len(ids) for cls, ids in unique_ids.items()}
    return final_counts, final_path

def normalize_region_points(region_points):
    """Normalize region_points: convert lists to tuples (JSON gives lists)"""
    if not region_points:
        return None
        
    if isinstance(region_points, dict):
        # Multi-region format: {"region-01": [(x,y), ...], "region-02": [...]}
        normalized = {}
        for key, pts in region_points.items():
            normalized[key] = [tuple(p) if isinstance(p, list) else p for p in pts]
        return normalized
    elif isinstance(region_points, (list, tuple)):
        # Single region format: [(x,y), ...]
        return [tuple(p) if isinstance(p, list) else p for p in region_points]
    return region_points

def video_polygon_cross_count(video_file: str, config: dict) -> tuple[dict, str]:
    """
    Count objects crossing regions or lines in a video and return:
    - dict of cumulative region counts
    - filepath to annotated video
    """
    # --- Configuration ---
    model_name = config.get("model", "yolo11n.pt")
    model_path = get_model_path(model_name)
    conf_threshold = config.get("conf_threshold", 0.5)
    classes = config.get("classes")
    
    # Normalize region_points to a dictionary
    raw_region_points = config.get("region_points")
    if isinstance(raw_region_points, dict):
        region_dict = raw_region_points
    elif isinstance(raw_region_points, (list, tuple)):
        region_dict = {"region-01": raw_region_points}
    else:
        # Default fallback
        region_dict = {}

    # Normalize point formats (ensure tuples)
    for name, pts in region_dict.items():
        region_dict[name] = [tuple(p) if isinstance(p, list) else p for p in pts]

    # --- Setup Video ---
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Error reading video file"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    base, ext = os.path.splitext(video_file)
    save_path = config.get("save_path", f"{base}_annotated.mp4")
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --- Tracking State ---
    model = YOLO(model_path)
    # name -> {track_id: class_name}
    cumulative_tracked_objects = {name: {} for name in region_dict}
    track_history = {} # (track_id, region_name) -> side (for lines)
    
    # Pre-defined colors for regions
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 165, 0), (128, 0, 128)
    ]

    def get_side(point, line_start, line_end):
        """Cross product to find which side of a line a point is on."""
        return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - \
               (line_end[1] - line_start[1]) * (point[0] - line_start[0]) > 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run multi-object tracking
        results = model.track(frame, persist=True, conf=conf_threshold, classes=classes, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            classes_idx = results[0].boxes.cls.int().cpu().numpy()
            names = results[0].names

            for tid, box, cidx in zip(ids, boxes, classes_idx):
                center = (int(box[0]), int(box[1]))
                class_name = names[cidx]
                
                for name, pts in region_dict.items():
                    if len(pts) > 2:
                        # --- Polygon Zone Counting ---
                        dist = cv2.pointPolygonTest(np.array(pts, dtype=np.int32), center, False)
                        if dist >= 0:
                            cumulative_tracked_objects[name][tid] = class_name
                    elif len(pts) == 2:
                        # --- Line Crossing Counting ---
                        line_start, line_end = pts[0], pts[1]
                        current_side = get_side(center, line_start, line_end)
                        key = (tid, name)
                        
                        if key in track_history:
                            prev_side = track_history[key]
                            if current_side != prev_side:
                                cumulative_tracked_objects[name][tid] = class_name
                        
                        track_history[key] = current_side

        # --- Draw Regions and Counts on Frame ---
        for i, (name, pts) in enumerate(region_dict.items()):
            color = colors[i % len(colors)]
            count = len(cumulative_tracked_objects[name])
            
            # Draw shape
            pts_array = np.array(pts, dtype=np.int32)
            if len(pts) > 2:
                cv2.polylines(annotated_frame, [pts_array], isClosed=True, color=color, thickness=2)
            else:
                cv2.line(annotated_frame, pts[0], pts[1], color=color, thickness=3)

            # Draw Label and Count
            label = f"{name}: {count}"
            text_pos = pts[0]
            cv2.putText(annotated_frame, label, (text_pos[0], text_pos[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        video_writer.write(annotated_frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Final results: breakdown by class per region
    final_counts = {}
    for name, tracked_objs in cumulative_tracked_objects.items():
        # tracked_objs is {tid: class_name}
        class_counts = Counter(tracked_objs.values())
        final_counts[name] = dict(class_counts)
        
    print(f"Cumulative counts: {final_counts}")

    # Convert to H.264
    base, ext = os.path.splitext(save_path)
    h264_path = f"{base}_h264.mp4"
    if convert_to_h264(save_path, h264_path):
        os.remove(save_path)
        final_path = h264_path
    else:
        final_path = save_path

    return final_counts, final_path

def image_zone_count(image_file: str, config: dict) -> tuple[dict, str]:
    """
    Count objects in defined regions of an image and return class-wise breakdown.
    """
    # --- Load image ---
    image = cv2.imread(image_file)
    assert image is not None, "Could not read image file"

    # --- Config ---
    model_name = config.get("model", "yolo11n.pt")
    model_path = get_model_path(model_name)
    conf = config.get("conf_threshold", 0.5)
    classes = config.get("classes")
    
    # Normalize region_points to dict
    raw_region_points = config.get("region_points")
    if isinstance(raw_region_points, dict):
        region_dict = raw_region_points
    elif isinstance(raw_region_points, (list, tuple)):
        region_dict = {"zone-01": raw_region_points}
    else:
        region_dict = {}

    # Normalize point formats
    for name, pts in region_dict.items():
        region_dict[name] = [tuple(p) if isinstance(p, list) else p for p in pts]

    # --- Run Inference ---
    model = YOLO(model_path)
    results = model.predict(image, conf=conf, classes=classes)
    annotated_image = results[0].plot()

    zone_counts = {name: Counter() for name in region_dict}

    if results[0].boxes:
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes_idx = results[0].boxes.cls.int().cpu().numpy()
        names = results[0].names

        for box, cidx in zip(boxes, classes_idx):
            center = (int(box[0]), int(box[1]))
            class_name = names[cidx]
            
            for name, pts in region_dict.items():
                if len(pts) >= 3:
                    dist = cv2.pointPolygonTest(np.array(pts, dtype=np.int32), center, False)
                    if dist >= 0:
                        zone_counts[name][class_name] += 1

    # Finalize colors and drawing for zones
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (name, pts) in enumerate(region_dict.items()):
        color = colors[i % len(colors)]
        pts_array = np.array(pts, dtype=np.int32)
        cv2.polylines(annotated_image, [pts_array], isClosed=True, color=color, thickness=2)
        
        total = sum(zone_counts[name].values())
        cv2.putText(annotated_image, f"{name}: {total}", (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- Save annotated image ---
    out_path = save_image(image_file, annotated_image)

    # Convert Counter to dict
    final_counts = {name: dict(counts) for name, counts in zone_counts.items()}
    return final_counts, out_path

def video_heatmap(video_file: str, config: dict) -> tuple[dict, str]:
    model_name = config.get("model")
    model_path = get_model_path(model_name)
    classes = config.get("classes")
    conf = config.get("conf_threshold")
    region_points = normalize_region_points(config.get("region_points"))
    
    base, ext = os.path.splitext(video_file)
    save_path = config.get("save_path", f"{base}_annotated.mp4")
    
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Error reading video file"



    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    # Initialize heatmap object
    heatmap = solutions.Heatmap(
        show=True,
        model=model_path,
        colormap=cv2.COLORMAP_PARULA,
        conf=conf,
        region=region_points,
        classes=classes,
    )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is empty or processing is complete.")
            break

        results = heatmap(im0)


        video_writer.write(results.plot_im)  # write the processed frame.

    # Convert to H.264 for browser compatibility
    base, ext = os.path.splitext(save_path)
    h264_path = f"{base}_h264.mp4"
    
    if convert_to_h264(save_path, h264_path):
        # Conversion successful, remove the mp4v version and use H.264
        os.remove(save_path)
        final_path = h264_path
    else:
        # Conversion failed, use the mp4v version (user can download it)
        print("Warning: H.264 conversion failed, using mp4v format")
        final_path = save_path
        
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # destroy all opened windows

    return results, final_path

def image_custom_classes(image_file: str, config: dict) -> tuple[dict, str]:
    """
    Count objects (using custom classes) in an image and save an annotated copy
    using the same naming convention as basic_count.
    """
    classes = config.get("classes")
    model_name = config.get("model")
    model_path = get_model_path(model_name)
    conf = config.get("conf_threshold")
    
    model = YOLOE(model_path)

    model.set_classes(classes)

    # Perform inference
    results = model.predict(image_file, conf=conf)

    # Convert YOLO results → Supervision Detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Build labels like "person", "car", etc.
    labels = [model.names[int(cls_id)] for cls_id in detections.class_id]

    # Annotate
    image = cv2.imread(image_file)
    box_annotator = sv.BoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.2, text_thickness=1)

    annotated = box_annotator.annotate(image.copy(), detections=detections)
    #annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)

    # Save annotated image
    out_path = save_image(image_file, annotated)

    # Count occurrences of each class
    object_counts = Counter(labels)

    return dict(object_counts), out_path

def video_custom_classes(video_path: str, config: dict) -> tuple[dict, str]:
    model_name = config.get("model")
    model_path = get_model_path(model_name)
    classes = config.get("classes")
    tracker = config.get("tracker", "botsort.yaml")
    conf_threshold = config.get("conf_threshold")

    model = YOLOE(model_path)

    model.set_classes(classes)

    # Prepare output path
    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_annotated.mp4"  # Force .mp4 extension

    # Grab video metadata for writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v codec (most reliable on Windows)
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.8,
        text_thickness=2,
        text_color=sv.Color.WHITE
    )

    # YOLO with tracking
    results_stream = model.track(
        source=video_path,
        tracker="botsort.yaml",
        stream=True,
        conf=conf_threshold
    )

    seen_ids = {}
    class_names = model.names

    for frame_results in results_stream:
        frame = frame_results.orig_img.copy()

        # Convert YOLO result → Supervision Detections
        det = sv.Detections.from_ultralytics(frame_results)

        if det.tracker_id is None or len(det) == 0:
            writer.write(frame)
            continue

        # Only keep detections with assigned track IDs
        det = det[det.tracker_id != None]

        # Generate labels like "person #12"
        labels = []
        for cls_id, track_id in zip(det.class_id, det.tracker_id):
            cls_name = class_names[cls_id]
            labels.append(f"{cls_name} #{track_id}")

            # Track unique IDs per class
            seen_ids.setdefault(cls_name, set()).add(int(track_id))

        # Annotate
        annotated = box_annotator.annotate(scene=frame, detections=det)
        annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)

        writer.write(annotated)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    # Small delay to ensure file is fully written and unlocked on Windows
    time.sleep(0.5)

    # Convert to H.264 for browser compatibility
    base, ext = os.path.splitext(out_path)
    h264_path = f"{base}_h264.mp4"
    
    if convert_to_h264(out_path, h264_path):
        # Conversion successful, remove the mp4v version and use H.264
        os.remove(out_path)
        final_path = h264_path
    else:
        # Conversion failed, use the mp4v version (user can download it)
        print("Warning: H.264 conversion failed, using mp4v format")
        final_path = out_path

    count_dict = {cls: len(ids) for cls, ids in seen_ids.items()}
    return count_dict, final_path
    
def test():

    config = Config(
        model="yoloe-11s-seg.pt",
        classes=['animal', 'human', 'hat'],
        tracker="botsort.yaml",
        conf_threshold=0.2,
        slice_wh=(960, 960),
        overlap_wh=(10, 10),
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        #region_points=[(100, 100), (100, 500), (500, 500), (500, 100)],
    )
    
    # results = image_custom_classes("horse.png", config)
    # print(f'YOLOE: {results}')


    complexity = estimate_image_complexity("horse.png", config)
    print(f'Complexity: {complexity}')

if __name__ == "__main__":
    test()
