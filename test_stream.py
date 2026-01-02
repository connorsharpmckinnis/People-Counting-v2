from functions import stream_count
import os

if __name__ == "__main__":
    # Example YouTube livestream URL (replace with a live one if this is dead)
    # This is a random city live cam for testing
    youtube_url = "https://www.youtube.com/watch?v=1-iS7LArMPA" # Update this to a live stream
    
    config = {
        "model": "yolo11n.pt",
        "conf_threshold": 0.25,
        "classes": [0, 2, 3, 5, 7] # person, car, motorcycle, bus, truck (COCO indices)
    }
    
    print(f"Starting stream count for {youtube_url}...")
    try:
        counts = stream_count(youtube_url, config)
        print(f"Final Counts: {counts}")
    except Exception as e:
        print(f"An error occurred: {e}")
