
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from collections import Counter
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from collections import defaultdict
import cv2
import numpy as np


def basic_count(image_path: str) -> dict:
    """
    Performs object detection using a YOLOv8n model and returns a dictionary
    of the counts of the detected objects.

    Args:
        image_path: The path to the input image.

    Returns:
        A dictionary where keys are object class names and values are their counts.
    """
    # Load a pre-trained YOLOv8n model (assuming 'yolo11n' was a conceptual request
    # and using a current, widely adopted YOLO model like YOLOv8n for implementation)
    model = YOLO('yolo11n.pt')

    # Perform inference on the image
    results = model(image_path)

    detected_classes = []
    for r in results:
        # Get class names from the model
        class_names = model.names
        # Extract detected class IDs and map them to names
        for cls_id in r.boxes.cls:
            detected_classes.append(class_names[int(cls_id)])

    # Count occurrences of each detected class
    object_counts = Counter(detected_classes)

    return dict(object_counts)

def sliced_count(image_path: str):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11s.pt",  # any yolov8/yolov9/yolo11/yolo12/rt-detr det model is supported
        confidence_threshold=0.35,
        device="cpu",  # or 'cuda:0' if GPU is available
    )
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    counts = Counter(pred.category.name for pred in result.object_prediction_list)

    return dict(counts)

def video_count(video_path: str = 0):
    model = YOLO("yolo11n.pt")

    # stream=True gives you a generator of per-frame results
    results_generator = model.track(
        source=video_path,
        show=True,
        tracker="bytetrack.yaml",
        stream=True
    )

    # Dictionary that maps class_name -> set of unique track IDs
    seen_ids = {}

    for frame_results in results_generator:
        class_names = model.names

        boxes = frame_results.boxes
        if boxes is None:
            continue

        # YOLOByteTrack gives each tracked box an `id`
        for box in boxes:
            if box.id is None:
                # Tracker hasn't assigned an ID yet
                continue

            cls_name = class_names[int(box.cls)]
            track_id = int(box.id)

            # Initialize the set for this class if needed
            if cls_name not in seen_ids:
                seen_ids[cls_name] = set()

            # Add the unique track ID for this class
            seen_ids[cls_name].add(track_id)

    # Convert sets → counts
    return {cls: len(ids) for cls, ids in seen_ids.items()}


def main():
    print("Hello from people-counting-v2!")
    #people = basic_count("image.png")
    #print(people)
    #people = sliced_count("image.png")
    #print(people)
    counts = sliced_video_count()
    print(counts)




if __name__ == "__main__":
    main()

