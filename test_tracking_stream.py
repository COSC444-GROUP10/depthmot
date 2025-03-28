import cv2
import numpy as np
import subprocess
from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sort import Sort

def main():
    # Initialize models
    print("Initializing models...")
    depth_estimator = MiDaSDepthEstimator(model_type="MiDaS_small")
    object_detector = YOLOv5Detector(model_name="yolov5s", conf_threshold=0.3)
    fusion = DepthDetectionFusion(depth_estimator, object_detector, depth_processing_interval=3)
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    
    # Stream URL
    stream_url = "https://streamserve.ok.ubc.ca/LiveCams/timcam.stream_720p/playlist.m3u8"
    
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
    
    print(f"Fetching stream from {stream_url}... Press 'q' to quit.")
    
    while True:
        # Read raw frame data from FFmpeg
        raw_frame = process.stdout.read(width * height * 3)  # 3 bytes per pixel (BGR)
        
        # If no data is received, the stream might have ended or failed
        if not raw_frame:
            print("Stream ended or failed to fetch data.")
            break

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