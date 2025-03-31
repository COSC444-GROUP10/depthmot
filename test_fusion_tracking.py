import cv2
import time
import numpy as np
import os
from pathlib import Path

from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sortIOU import SortIOU

# ======= CONFIGURATION =======
# Change these settings as needed
VIDEO_PATH = "data/viratSample.mp4"  # Path to the video file
DEPTH_INTERVAL = 1                   # Process depth every N frames
SAVE_FRAMES = False                  # Save individual frames as images (disabled by default)
OUTPUT_PATH = None                   # Path to save output video (set to None to disable)
DETECT_ONLY_PEOPLE = True            # Only detect people (class 0 in COCO dataset)
# Using standard visualization (not 3D)
SHOW_DEPTH = False                   # Show depth map alongside tracking visualization
# ============================

# SORT Tracker parameters
MAX_AGE = 50                        # Maximum number of frames to keep a track alive without matching
MIN_HITS = 3                         # Minimum number of hits to start a track
IOU_THRESHOLD = 0.3                 # Minimum IoU for a valid match

def test_video_with_tracking():
    """
    Test the fusion implementation with a video and SORT tracking
    
    Note: The SORT tracker has been modified to:
    1. Use 6D bounding boxes with depths [xmin, ymin, zmin, xmax, ymax, zmax]
    2. Include track IDs for each tracked object
    3. Return 7 values per track: [xmin, ymin, zmin, xmax, ymax, zmax, track_id]
    """
    print("Initializing models...")
    
    # Initialize models
    try:
        depth_estimator = MiDaSDepthEstimator(model_type="MiDaS_small")
        
        # Configure object detector to only detect people if specified
        object_detector = YOLOv5Detector(
            model_name="yolov8s.pt", 
            conf_threshold=0.3,
            classes=[0] if DETECT_ONLY_PEOPLE else None  # Class 0 is person in COCO dataset
        )
        
        fusion = DepthDetectionFusion(depth_estimator, object_detector, depth_processing_interval=DEPTH_INTERVAL)
        
        # Initialize SORT tracker
        tracker = SortIOU(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
        
        print(f"Depth estimation will run every {DEPTH_INTERVAL} frames")
        if DETECT_ONLY_PEOPLE:
            print("Detection limited to people only")
        
        # Use standard visualization
        show_depth = SHOW_DEPTH
        print(f"Standard visualization: Always ON")
        print(f"Depth map: {'ON' if show_depth else 'OFF'}")
        
        # Open video capture
        print(f"Opening video file: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {VIDEO_PATH}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Create output directory for frames if needed
        if SAVE_FRAMES:
            frames_dir = "output_frames"
            os.makedirs(frames_dir, exist_ok=True)
            print(f"Saving frames to {frames_dir}/")
        
        # Initialize video writer if output path is provided
        writer = None
        if OUTPUT_PATH:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_width = width * 2 if show_depth else width
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_width, height))
            print(f"Saving output to: {OUTPUT_PATH}")
        
        # Process frames
        frame_count = 0
        processing_times = []
        
        print("Processing video... Press 'q' to quit, 's' to save current frame, 'd' to toggle depth map")
        
        # Create a named window with normal size
        cv2.namedWindow('DepthMOT', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('DepthMOT', 1280, 720)  # Set a reasonable size
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            # Process frame
            start_time = time.time()
            
            # Process the frame
            if frame_count % DEPTH_INTERVAL == 0:
                # Process depth on this frame
                descriptors = fusion.process_frame(frame)
            else:
                # Skip depth processing on this frame
                descriptors = fusion.process_frame(frame)
            
            # Update tracker with current detections
            if descriptors:
                detections = fusion.get_sort_detections(descriptors)
                tracked_objects = tracker.update(detections)
            else:
                tracked_objects = tracker.update()
            
            # Create visualization
            vis_frame = frame.copy()
            
            # Draw detections with green bounding boxes
            if descriptors:
                for desc in descriptors:
                    x1, y1, x2, y2 = [int(coord) for coord in desc['bbox']]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw tracks with red center dots and ID/depth labels
            if tracked_objects is not None and len(tracked_objects) > 0:
                for track in tracked_objects:
                    # Format: [xmin, ymin, zmin, xmax, ymax, zmax, track_id]
                    x1, y1, z1, x2, y2, z2, track_id = track
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    center_z = (z1 + z2) / 2
                    
                    # Draw red center dot
                    cv2.circle(vis_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                    
                    # Draw track ID at top center of bounding box
                    label = f"ID:{int(track_id)}"
                    label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                    label_x = int((x1 + x2)/2 - label_width/2)
                    cv2.putText(vis_frame, label, (label_x, int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Print progress every 10 frames
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}/{total_frames}, {len(descriptors)} detections, {len(tracked_objects)} tracks, {1.0/processing_time:.2f} FPS")
            
            # Write frame to output video
            if writer:
                writer.write(vis_frame)
            
            # Save individual frame if requested
            if SAVE_FRAMES:
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, vis_frame)
            
            # Display frame
            cv2.imshow('DepthMOT', vis_frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User interrupted processing")
                break
            elif key == ord('s'):
                # Save current frame on 's' key press
                save_path = f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(save_path, vis_frame)
                print(f"Saved current frame to {save_path}")
            elif key == ord('d'):
                # Toggle depth map
                show_depth = not show_depth
                print(f"Depth map: {'ON' if show_depth else 'OFF'}")
                # Recreate video writer with new dimensions if necessary
                if writer:
                    writer.release()
                    out_width = width * 2 if show_depth else width
                    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (out_width, height))
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            avg_fps = 1.0 / avg_time
            print(f"Average processing time: {avg_time:.4f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total frames processed: {frame_count}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_with_tracking() 