import cv2
import time
import numpy as np
import os
import argparse
import sys
import subprocess

# Add the project root to sys.path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Use absolute imports from project root
from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sortEuclidean import SortEuclidean

# Default video path (adjust for new location)
DEFAULT_VIDEO_PATH = "data/viratSample.mp4"

def run_centroid_streaming(stream_url=None, record=False, detect_only_people=True):
    """
    Run the centroid-based tracking pipeline on a video stream
    
    Args:
        stream_url: URL to the video stream (None for webcam)
        record: Whether to record output video
        detect_only_people: Only detect people (class 0)
    """
    # Use default stream URL if none provided
    if stream_url is None:
        stream_url = "https://streamserve.ok.ubc.ca/LiveCams/timcam.stream_720p/playlist.m3u8"
        print(f"No stream URL provided, using default: {stream_url}")
    
    # Initialize models
    print("Initializing models...")
    
    # Initialize depth estimator
    depth_model = "MiDaS_small"
    depth_estimator = MiDaSDepthEstimator(model_type=depth_model)
    print(f"Depth estimator initialized with model: {depth_model}")
    
    # Initialize object detector
    detector_model = "yolov8s"
    detector_confidence = 0.35
    detector_classes = [0] if detect_only_people else None  # Only people (class 0) if specified
    detector = YOLOv5Detector(
        model_name=detector_model, 
        conf_threshold=detector_confidence,
        classes=detector_classes
    )
    print(f"Object detector initialized with model: {detector_model}, confidence: {detector_confidence}")
    if detect_only_people:
        print("Detecting only people (class 0)")
    
    # Initialize fusion module
    fusion = DepthDetectionFusion(depth_estimator, detector, depth_processing_interval=3)
    print("Fusion module initialized")
    
    # Initialize tracker
    max_age = 50
    min_hits = 3
    distance_threshold = 50.0
    tracker = SortEuclidean(max_age=max_age, min_hits=min_hits, distance_threshold=distance_threshold)
    print(f"SORT tracker initialized with max_age={max_age}, min_hits={min_hits}, distance_threshold={distance_threshold}")
    
    # Set the resolution for 720p (1280x720)
    width, height = 1280, 720
    fps = 30  # Assume 30 FPS for streams
    
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
    try:
        print(f"Fetching stream from {stream_url}...")
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Error starting FFmpeg: {e}")
        return
    
    print(f"Stream properties: {width}x{height} @ approx. {fps}fps")
    
    # Setup output video recorder if needed
    out = None
    if record:
        output_path = "centroid_streaming_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording output to: {output_path}")
    
    # Initialize parameters that can be modified during runtime
    depth_scale_factor = 1.0
    show_depth_map = False
    
    print("Starting tracking pipeline. Press 'q' to quit, 'd' to toggle depth map, '+/-' to adjust depth scale, 's' to save frame")
    
    # Initialize performance metrics
    frame_count = 0
    total_time = 0
    start_time = time.time()
    fps_update_interval = 10  # Update FPS every 10 frames
    current_fps = 0
    
    # Initialize storage for detection sizes
    detection_sizes = []
    
    try:
        while True:
            process_start = time.time()
            
            # Read raw frame data from FFmpeg
            raw_frame = process.stdout.read(width * height * 3)  # 3 bytes per pixel (BGR)
            
            # If no data is received, the stream might have ended or failed
            if not raw_frame or len(raw_frame) != width * height * 3:
                print("Stream ended or failed to fetch data.")
                break
            
            # Convert raw bytes to a numpy array and reshape into a frame
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            
            frame_count += 1
            
            # Process frame
            depth_estimator.depth_scale_factor = depth_scale_factor
            descriptors = fusion.process_frame(frame)
            
            # Update detection sizes for better bbox estimation
            if descriptors:
                for desc in descriptors:
                    x1, y1, x2, y2 = desc['bbox']
                    w, h = x2 - x1, y2 - y1
                    detection_sizes.append((w, h))
                    # Keep only the most recent 100 detections
                    if len(detection_sizes) > 100:
                        detection_sizes.pop(0)
            
            # Convert to SORT format and update tracker
            if descriptors:
                detections = fusion.get_sort_detections(descriptors)
                tracks = tracker.update(detections)
            else:
                tracks = tracker.update()
            
            # Calculate processing time
            process_time = time.time() - process_start
            total_time += process_time
            
            # Update FPS calculation periodically
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                current_fps = fps_update_interval / elapsed if elapsed > 0 else 0
                start_time = time.time()
            
            # Calculate average detection size for bbox estimation
            avg_width, avg_height = 80, 160  # Default values
            if detection_sizes:
                avg_width = sum(w for w, h in detection_sizes) / len(detection_sizes)
                avg_height = sum(h for w, h in detection_sizes) / len(detection_sizes)
            
            # Visualize results
            if show_depth_map:
                # Show depth map visualization
                depth_map = depth_estimator.last_depth_map.copy()
                depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                vis_frame = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_PLASMA)
                
                # Add text indicating it's the depth map
                cv2.putText(vis_frame, "Depth Map", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
            else:
                # Show tracking visualization
                vis_frame = frame.copy()
                
                # Draw detections (with green bounding boxes)
                if descriptors:
                    for desc in descriptors:
                        x1, y1, x2, y2 = [int(coord) for coord in desc['bbox']]
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                # Draw tracks (with red center dots only)
                if tracks is not None and tracks.shape[0] > 0:
                    for track in tracks:
                        # Centroid coordinates and ID
                        center_x, center_y, center_z, track_id = track
                        
                        # Draw center point as red dot
                        cv2.circle(vis_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                        
                        # Calculate a bounding box around the centroid using average detection sizes
                        x1 = int(center_x - avg_width/2)
                        y1 = int(center_y - avg_height/2)
                        x2 = int(center_x + avg_width/2)
                        y2 = int(center_y + avg_height/2)
                        
                        # Draw track ID at top center of bounding box
                        label = f"ID:{int(track_id)}"
                        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                        label_x = int(center_x - label_width/2)
                        cv2.putText(vis_frame, label, (label_x, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            # Display frame
            cv2.imshow("Centroid Streaming", vis_frame)
            
            # Record frame if needed
            if out is not None:
                out.write(vis_frame)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User terminated")
                break
            elif key == ord('d'):
                show_depth_map = not show_depth_map
                print(f"Depth map display: {'ON' if show_depth_map else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                depth_scale_factor += 0.1
                print(f"Depth scale factor increased to {depth_scale_factor:.2f}")
            elif key == ord('-'):
                depth_scale_factor = max(0.1, depth_scale_factor - 0.1)
                print(f"Depth scale factor decreased to {depth_scale_factor:.2f}")
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = f"centroid_streaming_frame_{timestamp}.jpg"
                cv2.imwrite(save_path, vis_frame)
                print(f"Frame saved to {save_path}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    # Print performance statistics
    if frame_count > 0:
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
    
    # Clean up resources
    process.stdout.close()
    process.stderr.close()
    process.terminate()
        
    if out is not None:
        out.release()
        
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centroid-based Tracking for Streaming")
    
    parser.add_argument("--stream", type=str, default=None, 
                      help="URL to the video stream (default: UBC live stream)")
    parser.add_argument("--record", action="store_true", 
                      help="Record output video")
    parser.add_argument("--detect-all", action="store_true", 
                      help="Detect all objects (default: only people)")
    
    args = parser.parse_args()
    
    run_centroid_streaming(
        stream_url=args.stream,
        record=args.record,
        detect_only_people=not args.detect_all
    ) 