import cv2
import numpy as np
import subprocess
from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sort import Sort
import time

# ======= CONFIGURATION =======
# Change these settings as needed
VIDEO_PATH = "data/042.mp4"  # Path to the video file
DEPTH_INTERVAL = 1                   # Process depth every N frames
SAVE_FRAMES = False                  # Save individual frames as images (disabled by default)
OUTPUT_PATH = None                   # Path to save output video (set to None to disable)
DETECT_ONLY_PEOPLE = True            # Only detect people (class 0 in COCO dataset)
DEPTH_MODEL = "DPT_Large"              # MiDaS model type (options: 'MiDaS_small', 'DPT_Large', 'DPT_Hybrid')
# ============================

# SORT Tracker parameters
MAX_AGE = 100                          # Maximum number of frames to keep a track alive without matching
MIN_HITS = 5                         # Minimum number of hits to start a track
IOU_THRESHOLD = 0.1                  # Minimum IoU for a valid match

def main():
    # Initialize models
    print("Initializing models...")
    depth_estimator = MiDaSDepthEstimator(model_type=DEPTH_MODEL)
    # Configure object detector to only detect people if specified
    object_detector = YOLOv5Detector(
            model_name="yolov8s.pt", 
            conf_threshold=0.3,
            classes=[0] if DETECT_ONLY_PEOPLE else None  # Class 0 is person in COCO dataset
        )
    fusion = DepthDetectionFusion(depth_estimator, object_detector, depth_processing_interval=3)
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    
    # Stream URL
    stream_url = "https://streamserve.ok.ubc.ca/LiveCams/timcam.stream_720p/playlist.m3u8" # 3.5 FPS
    
    # FFmpeg command to fetch the stream and pipe raw video frames
    ffmpeg_command = [
        "ffmpeg",
        "-i", stream_url,           # Input stream URL
        "-f", "rawvideo",          # Output format as raw video
        "-pix_fmt", "bgr24",       # Pixel format compatible with OpenCV (BGR)
        "-an",                     # No audio
        "pipe:"                    # Pipe output to stdout
    ]
    
    # Start FFmpeg process
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Set the resolution for 720p (1280x720)
    width, height = 1280, 720
    fps = 3.5  # Adjust this based on the actual FPS of the stream
    frame_delay = 1.0 / fps  # Time delay to match FPS
    
    print(f"Fetching stream from {stream_url} at {fps} FPS... Press 'q' to quit.")
    
    while True:
        start_time = time.time()  # Track frame start time
        # Read raw frame data from FFmpeg
        raw_frame = process.stdout.read(width * height * 3)  # 3 bytes per pixel (BGR)
        
         # If no data is received, stream might be buffering or down
        if not raw_frame:
            print("Stream ended or buffering...")
            time.sleep(0.5)  # Wait before retrying
            continue

        # Convert raw bytes to a numpy array and reshape into a frame
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        
        # Process frame using fusion module
        descriptors = fusion.process_frame(frame)
        
        # Get SORT detections and update tracker
        if descriptors:
            detections = fusion.get_sort_detections(descriptors)
            tracked_objects = tracker.update(detections)
            
            # Visualize results with tracking
            vis_frame = fusion.visualize_with_tracking(frame, descriptors, tracked_objects)
        else:
            vis_frame = fusion.visualize(frame, descriptors)
        
        # Display the frame
        cv2.imshow("DepthMOT Stream", vis_frame)

        # Adjust loop timing to match FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed_time)
        time.sleep(sleep_time)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    process.stdout.close()
    process.stderr.close()
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == "__main__":
    main() 