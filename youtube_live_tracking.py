import cv2
import yt_dlp
import time
import numpy as np
from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sort import Sort

# ======= CONFIGURATION =======
YOUTUBE_URL = "https://www.youtube.com/watch?v=DLmn7f9SJ5A"  # Replace with actual YouTube Live ID
DEPTH_INTERVAL = 10
SAVE_FRAMES = False
OUTPUT_PATH = None
DETECT_ONLY_PEOPLE = True
DEPTH_MODEL = "DPT_Large"  # MiDaS model type (options: 'MiDaS_small', 'DPT_Large', 'DPT_Hybrid')
TARGET_FPS = 30  # Limit FPS to prevent lag
# ============================

# SORT Tracker parameters
MAX_AGE = 100
MIN_HITS = 5
IOU_THRESHOLD = 0.1

def get_youtube_stream_url(youtube_url):
    """Extracts the .m3u8 stream URL from a YouTube Live video using yt-dlp."""
    ydl_opts = {"quiet": True, "format": "best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]  # Extract direct stream URL

def main():
    print("Initializing models...")
    depth_estimator = MiDaSDepthEstimator(model_type=DEPTH_MODEL)
    object_detector = YOLOv5Detector(
        model_name="yolov8s.pt", 
        conf_threshold=0.3,
        classes=[0] if DETECT_ONLY_PEOPLE else None
    )
    fusion = DepthDetectionFusion(depth_estimator, object_detector, depth_processing_interval=3)
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

    # Get YouTube live stream URL
    stream_url = get_youtube_stream_url(YOUTUBE_URL)
    print(f"Streaming from: {stream_url}")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return

    frame_delay = 1.0 / TARGET_FPS
    last_time = time.time()

    print(f"Fetching stream at {TARGET_FPS} FPS... Press 'q' to quit.")

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Stream ended or buffering...")
            time.sleep(0.5)
            continue

        # Skip frames to maintain target FPS
        if time.time() - last_time >= frame_delay:
            last_time = time.time()

            descriptors = fusion.process_frame(frame)

            if descriptors:
                detections = fusion.get_sort_detections(descriptors)
                tracked_objects = tracker.update(detections)
                vis_frame = fusion.visualize_with_tracking(frame, descriptors, tracked_objects)
            else:
                vis_frame = fusion.visualize(frame, descriptors)

            cv2.imshow("DepthMOT Stream", vis_frame)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)
        time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == "__main__":
    main()
