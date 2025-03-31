#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import argparse

from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sort2 import SortEuclidean

# Video path
VIDEO_PATH = "data/MOT17-02-SDP-raw.webm"

def run_visualization(max_frames=None, record=False):
    """
    Run a visualization that shows RGB frames with detections side by side with 
    depth maps showing the same bounding boxes
    
    Args:
        max_frames: Maximum number of frames to process (None for all)
        record: Whether to record output video
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
    fusion = DepthDetectionFusion(depth_estimator, detector, depth_processing_interval=1)
    print("Fusion module initialized with depth processing on every frame")
    
    # Initialize tracker
    max_age = 50
    min_hits = 3
    distance_threshold = 50.0
    tracker = SortEuclidean(max_age=max_age, min_hits=min_hits, distance_threshold=distance_threshold)
    print(f"SORT2 tracker initialized with max_age={max_age}, min_hits={min_hits}, distance_threshold={distance_threshold}")
    
    # Open video
    print(f"Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_PATH}")
        return
        
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS if not available
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    
    # Define output dimensions (side by side visualization)
    output_width = frame_width * 2  # RGB and depth side by side
    output_height = frame_height
    
    # Setup output video recorder if needed
    out = None
    if record:
        output_path = "depth_tracking_visualization.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        print(f"Recording output to: {output_path}")
    
    # Create output directory for saved frames
    output_dir = "output_depth_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Initialize parameters
    frame_count = 0
    total_time = 0
    font_scale = 0.7
    
    print("Starting visualization pipeline. Press 'q' to quit, '+/-' to adjust depth scale, 's' to save frame")
    
    while True:
        # Read frame from video
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
        
        # Create RGB visualization
        rgb_vis = frame.copy()
        
        # Create depth map visualization - raw output from MiDaS
        depth_map = fusion.last_depth_map.copy()
        
        # Convert the raw depth map to 8-bit grayscale for visualization
        # Not normalizing or scaling - just converting to appropriate data type
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        depth_vis_raw = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        # Convert to BGR for visualization
        depth_vis = cv2.cvtColor(depth_vis_raw, cv2.COLOR_GRAY2BGR)  # Convert to BGR for consistency
        
        # Draw tracks - only show these bounding boxes
        if tracks is not None and tracks.shape[0] > 0:
            for track in tracks:
                track_x, track_y, track_z, track_id = track
                
                # Find the closest detection to this track to use its bounding box
                closest_det = None
                min_dist = float('inf')
                
                if descriptors:
                    for desc in descriptors:
                        x1, y1, x2, y2 = desc['bbox']
                        det_center_x = (x1 + x2) / 2
                        det_center_y = (y1 + y2) / 2
                        
                        # Calculate distance to track
                        dist = np.sqrt((det_center_x - track_x)**2 + (det_center_y - track_y)**2)
                        
                        # Update closest detection if this one is closer
                        if dist < min_dist:
                            min_dist = dist
                            closest_det = desc
                
                # Use the found detection's bbox if we have one, otherwise use a default size
                if closest_det is not None:
                    x1, y1, x2, y2 = [int(coord) for coord in closest_det['bbox']]
                else:
                    # Fallback to default size around the track point
                    box_width, box_height = 100, 200
                    x1 = int(track_x - box_width / 2)
                    y1 = int(track_y - box_height / 2)
                    x2 = int(track_x + box_width / 2)
                    y2 = int(track_y + box_height / 2)
                
                # Draw on RGB frame - green box with red center
                cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(rgb_vis, (int(track_x), int(track_y)), 5, (0, 0, 255), -1)
                
                # Draw on depth frame - blue box only
                cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add only frame number to both frames
        cv2.putText(rgb_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), 2)
        cv2.putText(depth_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), 2)
        
        # Combine the two visualizations side by side
        combined_vis = np.hstack((rgb_vis, depth_vis))
        
        # Display combined visualization
        cv2.imshow("RGB and Depth Tracking", combined_vis)
        
        # Save frame to output directory
        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, combined_vis)
        
        # Record frame if needed
        if out is not None:
            out.write(combined_vis)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('+') or key == ord('='):
            depth_scale_factor *= 1.1
            print(f"Increased depth scale factor to {depth_scale_factor:.2f}")
        elif key == ord('-'):
            depth_scale_factor *= 0.9
            print(f"Decreased depth scale factor to {depth_scale_factor:.2f}")
        elif key == ord('s'):
            screenshot_path = f"depth_tracking_frame_{frame_count}.png"
            cv2.imwrite(screenshot_path, combined_vis)
            print(f"Saved screenshot to {screenshot_path}")
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    if frame_count > 0:
        avg_fps = frame_count / total_time
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
        print(f"Frames saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RGB frames with detections side by side with depth maps")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--record", action="store_true", help="Record output video")
    
    args = parser.parse_args()
    
    run_visualization(max_frames=args.max_frames, record=args.record) 