import multiprocessing
import traceback
import json
import os
import shutil
from pathlib import Path

from job_store import update_job_status


def worker_loop(queue: multiprocessing.Queue):
    """
    Main worker loop.
    Pulls job_ids from the queue and processes them.
    """
    print(f"Worker process started. PID: {os.getpid()}")

    # Lazy import inside spawned process (CUDA-safe)
    import functions

    # Mapping job types to function names
    JOB_HANDLERS = {
        "basic": functions.basic_count,
        "sliced": functions.sliced_count,
        "video": functions.video_count,
        "sliced_video": functions.sliced_video_count,
        "polygon_cross_count": functions.video_polygon_cross_count,
        "image_zone_count": functions.image_zone_count,
        "image_custom": functions.image_custom_classes,
        "video_custom": functions.video_custom_classes,
        "stream": functions.stream_count,
    }
    
    while True:
        try:
            # Blocking get
            job_id = queue.get()
            
            print(f"Worker picked up job: {job_id}")
            update_job_status(job_id, "processing")
            
            # Fetch job details handled inside process_job or passed?
            # We only passed ID to keep queue light. Let's fetch from DB.
            from job_store import get_job
            job = get_job(job_id)
            
            if not job:
                print(f"Job {job_id} not found in DB!")
                continue
                
            job_type = job["type"]
            filename = job["filename"]
            config = json.loads(job["config"])
            
            if job_type not in JOB_HANDLERS:
                raise ValueError(f"Unknown job type: {job_type}")
                
            handler = JOB_HANDLERS[job_type]
            
            # Run the actual processing
            print(f"Running {job_type} on {filename}...")
            counts, annotated_path = handler(filename, config=config)
            
            # Move result to final location if needed?
            # The functions currently return a path.
            # Some functions in endpoints.py were moving files to RESULTS_DIR.
            # We should probably standardize this. 
            # For now, let's assume the handler returns a path, and we might need to move it 
            # to a public 'results' folder if it's not already there by the function?
            # Looking at functions.py, most save to a derived path.
            # Endpoints.py was moving them. Let's replicate that logic here.
            
            RESULTS_DIR = Path("results")
            RESULTS_DIR.mkdir(exist_ok=True)
            
            ext = os.path.splitext(annotated_path)[1]
            final_filename = f"{job_id}{ext}"
            final_path = RESULTS_DIR / final_filename
            
            # Clean move
            if os.path.exists(annotated_path):
                shutil.move(annotated_path, final_path)
            
            # Result object to store in DB
            result_data = {
                "counts": counts,
                "annotated_file": f"/secure-results/{final_filename}",
                "file_type": "video" if ext.lower() == ".mp4" else "image"
            }
            
            update_job_status(job_id, "completed", result=result_data)
            print(f"Job {job_id} completed successfully.")
            
            # Clean up input file if it's a real file (not a URL)
            try:
                if os.path.exists(filename) and not filename.startswith("http"):
                     os.remove(filename)
            except:
                pass

        except Exception as e:
            print(f"Job failed: {e}")
            traceback.print_exc()
            update_job_status(job_id, "failed", error=str(e))

