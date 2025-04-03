import cv2
import time
import numpy as np
import os
import argparse
import sys

# Add the project root to sys.path to make imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Use absolute imports from project root
from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sortEuclidean import SortEuclidean

VIDEO_PATH = "data/viratSample.mp4"  # Path to the video file (adjust for new location)

def run_centroid_tracking(video_path=VIDEO_PATH, record=False, max_frames=None):
    """
    Run the centroid-based tracking pipeline on a video file or webcam
    
    Args:
        video_path: Path to video file (None for webcam)
        record: Whether to record output video
        max_frames: Maximum number of frames to process (None for all)
    """
    # Initialize models
    print("Initializing models...")
    
    # Initialize depth estimator
    depth_model = "MiDaS_small"
    depth_estimator = MiDaSDepthEstimator(model_type=depth_model)
    print(f"Depth estimator initialized with model: {depth_model}")
    
    # Initialize object detector
    detector_model = "yolov8s"
    detector_confidence = 0.35
    detector = YOLOv5Detector(model_name=detector_model, conf_threshold=detector_confidence)
    print(f"Object detector initialized with model: {detector_model}, confidence: {detector_confidence}")
    
    # Initialize fusion module
    fusion = DepthDetectionFusion(depth_estimator, detector, depth_processing_interval=3)
    print("Fusion module initialized")
    
    # Initialize tracker
    max_age = 50
    min_hits = 3
    distance_threshold = 50.0
    tracker = SortEuclidean(max_age=max_age, min_hits=min_hits, distance_threshold=distance_threshold)
    print(f"SORT tracker initialized with max_age={max_age}, min_hits={min_hits}, distance_threshold={distance_threshold}")
    
    # Open video source
    if video_path is None or video_path.lower() == "webcam":
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30  # Default FPS if not available
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    
    # Setup output video recorder if needed
    out = None
    if record:
        output_path = "centroid_tracking_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Recording output to: {output_path}")
    
    # Initialize performance metrics
    frame_count = 0
    total_time = 0
    
    # Initialize parameters that can be modified during runtime
    depth_scale_factor = 1.0
    show_depth = False
    font_scale = 0.7  # Font scale for all text
    
    print("Starting tracking pipeline. Press 'q' to quit, 'd' to toggle depth map, '+/-' to adjust depth scale, 's' to save frame")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        frame_count += 1
        if max_frames is not None and frame_count > max_frames:
            print(f"Reached maximum frame count: {max_frames}")
            break
        
        # Start timing
        start_time = time.time()
        
        # Process frame
        depth_estimator.depth_scale_factor = depth_scale_factor
        descriptors = fusion.process_frame(frame)
        
        # Convert to SORT format and update tracker
        if descriptors:
            detections = fusion.get_sort_detections(descriptors)
            tracks = tracker.update(detections)
        else:
            tracks = tracker.update()
        
        # End timing
        process_time = time.time() - start_time
        total_time += process_time
        
        # Visualize results
        if show_depth:
            # Show depth map
            depth_map = depth_estimator.last_depth_map.copy()
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = depth_map_normalized
            cv2.imshow("Depth Map", depth_colormap)
        else:
            # Show tracking results
            output_frame = frame.copy()
            
            # Draw detections with green bounding boxes
            if descriptors:
                for desc in descriptors:
                    x1, y1, x2, y2 = [int(coord) for coord in desc['bbox']]
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw tracks with red center dots and ID/depth labels
            if tracks is not None and len(tracks) > 0:
                for d in tracks:
                    # Get the centroid coordinates and ID
                    centroid_x, centroid_y, centroid_z, track_id = d
                    
                    # Draw the red center dot
                    cv2.circle(output_frame, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)
                    
                    # Add track ID label at the top of the bounding box
                    # Estimate a rough bounding box based on centroids and average sizes
                    box_width, box_height = 100, 200  # Default sizes
                    if descriptors:
                        widths = []
                        heights = []
                        for desc in descriptors:
                            x1, y1, x2, y2 = desc['bbox']
                            widths.append(x2 - x1)
                            heights.append(y2 - y1)
                        if widths and heights:
                            box_width = sum(widths) / len(widths)
                            box_height = sum(heights) / len(heights)
                    
                    # Calculate top center of the box
                    x1 = int(centroid_x - box_width/2)
                    y1 = int(centroid_y - box_height/2)
                    x2 = int(centroid_x + box_width/2)
                    y2 = int(centroid_y + box_height/2)
                    
                    # Draw ID label at top center of box
                    label = f"ID:{int(track_id)}"
                    label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                    label_x = int(centroid_x - label_width/2)
                    cv2.putText(output_frame, label, (label_x, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow("Centroid Tracking", output_frame)
        
        # Record frame if needed
        if out is not None:
            out.write(output_frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User terminated")
            break
        elif key == ord('d'):
            show_depth = not show_depth
            print(f"Depth map display: {'ON' if show_depth else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            depth_scale_factor += 0.1
            print(f"Depth scale factor increased to {depth_scale_factor:.2f}")
        elif key == ord('-'):
            depth_scale_factor = max(0.1, depth_scale_factor - 0.1)
            print(f"Depth scale factor decreased to {depth_scale_factor:.2f}")
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = f"centroid_tracking_frame_{timestamp}.jpg"
            cv2.imwrite(save_path, output_frame)
            print(f"Frame saved to {save_path}")
    
    # Print performance statistics
    if frame_count > 0:
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centroid-based Tracking Demo")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, 
                       help=f"Path to video file (default: {VIDEO_PATH}, 'webcam' for camera)")
    parser.add_argument("--record", action="store_true", 
                       help="Record output video")
    parser.add_argument("--max-frames", type=int, default=None, 
                       help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    run_centroid_tracking(
        video_path=args.video,
        record=args.record,
        max_frames=args.max_frames
    ) 