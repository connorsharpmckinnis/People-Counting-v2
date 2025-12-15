
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from collections import Counter
from ultralytics import YOLO, solutions
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


def basic_count(image_path: str, config: dict):
    model_path = config.get("model", "yolo11n.pt")
    conf_threshold = config.get("conf_threshold", 0.35)

    model = YOLO(model_path)

    # Perform inference
    results = model(image_path, conf=conf_threshold, classes=config.get("classes"))

    # Convert YOLO results → Supervision Detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Build labels like "person", "car", etc.
    labels = [model.names[int(cls_id)] for cls_id in detections.class_id]

    # Annotate
    image = cv2.imread(image_path)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    annotated = box_annotator.annotate(image.copy(), detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)

    # Save annotated image
    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_annotated{ext}"
    cv2.imwrite(out_path, annotated)

    # Count occurrences of each class
    object_counts = Counter(labels)

    return dict(object_counts), out_path

def sliced_count(image_path: str, config: dict):
    model = config.get("model", "yolo11n.pt")
    conf_threshold = config.get("conf_threshold", 0.35)
    slice_height = config.get("slice_height", 256)
    slice_width = config.get("slice_width", 256)
    overlap_height_ratio = config.get("overlap_height_ratio", 0.2)
    overlap_width_ratio = config.get("overlap_width_ratio", 0.2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model,
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

    base, ext = os.path.splitext(image_path)
    output_folder = f"{base}_annotated"      # folder SAHI will create
    final_path = f"{base}_annotated{ext}"    # final annotated image path

    # Filter by class if specified
    classes_to_track = config.get("classes")
    if classes_to_track is not None:
        # result.object_prediction_list contains ObjectPrediction items
        # checking against category.id (int)
        filtered_list = []
        for pred in result.object_prediction_list:
            if pred.category.id in classes_to_track:
                filtered_list.append(pred)
        result.object_prediction_list = filtered_list

    # export visuals to folder
    result.export_visuals(output_folder, text_size=int(1))

    # move prediction_visual.png → desired path
    shutil.move(
        os.path.join(output_folder, "prediction_visual.png"),
        final_path
    )

    # remove empty folder
    os.rmdir(output_folder)

    counts = Counter(pred.category.name for pred in result.object_prediction_list)

    return dict(counts), final_path

def video_count(video_path: str, config: dict):
    model_path = config.get("model", "yolo11n.pt")
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
):
    model_path = config.get("model", "yolo11n.pt")
    base, ext = os.path.splitext(video_path)
    save_path = f"{base}_annotated.mp4"  # Force .mp4 extension
    slice_wh = config.get("slice_wh", (960, 960))
    overlap_wh = config.get("overlap_wh", (10, 10))
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

def video_polygon_cross_count(video_file: str, config: dict):
    """
    Count objects crossing a line or polygon in a video and return:
    - dict of crossing counts
    - filepath to annotated video
    """
    base, ext = os.path.splitext(video_file)
    save_path = config.get("save_path", f"{base}_annotated.mp4")
    
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Error reading video file"

    # Video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    annotated_video_writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # Line counting (2 points = line), polygon counting (4 points = polygon) 
    # Use config points or default to a vertical line in the middle
    region_points = config.get("region_points")
    if not region_points:
        mid_x = w // 2
        region_points = [(mid_x, 0), (mid_x, h)]

    counter = solutions.ObjectCounter(
        show=False,
        region=region_points,
        model=config.get("model", "yolo11n.pt"),
        conf=config.get("conf_threshold", 0.5),
        classes=config.get("classes"),
        tracker=config.get("tracker", "bytetrack.yaml"),
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        results = counter(im0)
        annotated_video_writer.write(results.plot_im)

    cap.release()
    annotated_video_writer.release()
    cv2.destroyAllWindows()
    
    # Wait a bit for file to release
    time.sleep(0.5)

    # --- Extract counts from the counter ---
    cross_counts = getattr(counter, "classwise_count", {})

    print(cross_counts)
    
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

    return cross_counts, final_path

def image_zone_count(image_file: str, config: dict):
    """
    Count objects in a defined region of an image and save an annotated copy
    using the same naming convention as basic_count.
    """

    # --- Load image ---
    image = cv2.imread(image_file)
    assert image is not None, "Could not read image file"

    # --- Output path: same convention as basic_count ---
    base, ext = os.path.splitext(image_file)
    out_path = f"{base}_annotated{ext}"

    # --- Config ---
    model = config.get("model", "yolo11n.pt")
    conf = config.get("conf_threshold", 0.5)
    classes = config.get("classes")
    region_points = config.get("region_points")

    # --- Initialize region counter ---
    regioncounter = solutions.RegionCounter(
        show=False,
        region=region_points,
        model=model,
        conf=conf,
        classes=classes,
    )

    # --- Process image ---
    results: solutions.RegionCounterResult = regioncounter.process(image)

    # --- Save annotated image ---
    cv2.imwrite(out_path, results.plot_im)

    # --- Return counts + path ---
    return results.region_counts, out_path

def test():

    config = Config(
        model="yolo11l.pt",
        classes=[0],
        tracker="bytetrack.yaml",
        conf_threshold=0.5,
        slice_wh=(960, 960),
        overlap_wh=(10, 10),
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        region_points=[(100, 100), (100, 500), (500, 500), (500, 100)],
        output_path="test_image_annotated.jpg",
        input_path="parade.jpg"
    )
    image_zone_count("parade.jpg", config)


if __name__ == "__main__":
    test()
