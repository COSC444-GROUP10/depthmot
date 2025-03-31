import cv2
import time
import numpy as np
import subprocess
from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sortIOU import SortIOU

DETECT_ONLY_PEOPLE = True  

def run_tracking_stream():
    # Stream URL - UBC Live Stream 720p
    stream_url = "https://streamserve.ok.ubc.ca/LiveCams/timcam.stream_720p/playlist.m3u8"
    
    # Initialize models
    depth_estimator = MiDaSDepthEstimator(model_type="MiDaS_small")
    object_detector = YOLOv5Detector(
        model_name="yolov8s",
        conf_threshold=0.35,
        classes=[0] if DETECT_ONLY_PEOPLE else None
      )
    fusion = DepthDetectionFusion(depth_estimator, object_detector, depth_processing_interval=3)
    tracker = SortIOU(max_age=50, min_hits=3, iou_threshold=0.2)
    
    # Stream URL
    print(f"Connecting to stream: {stream_url}")
    
    # Set up FFmpeg command to fetch the stream (can work with RTSP, HLS, etc.)
    width, height = 1280, 720
    ffmpeg_command = [
        "ffmpeg",
        "-i", stream_url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "pipe:"
    ]
    
    # Start FFmpeg process
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        while True:
            # Read raw video frame from FFmpeg
            raw_image = process.stdout.read(width*height*3)
            if not raw_image:
                print("End of stream")
                break
            
            # Convert raw bytes to numpy array and reshape to OpenCV format
            frame = np.frombuffer(raw_image, np.uint8).reshape((height, width, 3))
            
            # Process frame through the fusion pipeline
            start_time = time.time()
            descriptors = fusion.process_frame(frame)
            
            # Convert descriptors to SortIOU format and update tracker
            if descriptors:
                tracking_dets = fusion.get_sort_detections(descriptors)
                tracks = tracker.update(tracking_dets)
            else:
                tracks = tracker.update(np.empty((0, 7)))
            
            # Create visualization
            output_frame = frame.copy()
            
            # Draw detections with green bounding boxes
            if descriptors:
                for desc in descriptors:
                    x1, y1, x2, y2 = [int(coord) for coord in desc['bbox']]
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw tracks with red center dots and ID/depth labels
            if tracks is not None and len(tracks) > 0:
                for track in tracks:
                    if len(track) >= 7:  # 6D bbox + ID format
                        x1, y1, z1, x2, y2, z2, track_id = track
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        center_z = (z1 + z2) / 2
                        
                        # Draw red center dot
                        cv2.circle(output_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                        
                        # Draw track ID at top center of bounding box
                        label = f"ID:{int(track_id)}"
                        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                        label_x = int((x1 + x2)/2 - label_width/2)
                        cv2.putText(output_frame, label, (label_x, int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display output
            cv2.imshow("Live Tracking", output_frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        process.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking_stream() 