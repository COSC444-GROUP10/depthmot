import cv2
import time
import numpy as np
import os
from pathlib import Path

from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion

# ======= CONFIGURATION =======
# Change these settings as needed
VIDEO_PATH = "data/viratSample.mp4"  # Path to the video file
DEPTH_INTERVAL = 5                   # Process depth every N frames
SAVE_FRAMES = False                  # Save individual frames as images (disabled by default)
OUTPUT_PATH = None                   # Path to save output video (set to None to disable)
DETECT_ONLY_PEOPLE = True            # Only detect people (class 0 in COCO dataset)
# ============================

def test_video():
    """
    Test the fusion implementation with a video
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
        
        print(f"Depth estimation will run every {DEPTH_INTERVAL} frames")
        if DETECT_ONLY_PEOPLE:
            print("Detection limited to people only")
        
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
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
            print(f"Saving output to: {OUTPUT_PATH}")
        
        # Process frames
        frame_count = 0
        processing_times = []
        
        print("Processing video... Press 'q' to quit, 's' to save current frame")
        
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
            
            # Get 3D descriptors
            descriptors = fusion.process_frame(frame)
            
            # Visualize results
            vis_frame = fusion.visualize(frame, descriptors)
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Display FPS
            fps_text = f"FPS: {1.0/processing_time:.2f}"
            cv2.putText(vis_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame count
            frame_count += 1
            count_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(vis_frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display number of detections
            det_text = f"Detections: {len(descriptors)}"
            cv2.putText(vis_frame, det_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Print progress every 10 frames
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}/{total_frames}, {len(descriptors)} detections, {1.0/processing_time:.2f} FPS")
            
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
    test_video() 