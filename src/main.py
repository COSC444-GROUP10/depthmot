import cv2
import argparse
import time
import numpy as np
from pathlib import Path

from depth.midas_depth import MiDaSDepthEstimator
from detection.yolo_detector import YOLOv5Detector
from fusion.descriptor_fusion import DepthDetectionFusion

def process_video(video_path, output_path=None, display=True, depth_interval=3):
    """
    Process a video file or camera stream
    
    Args:
        video_path (str): Path to video file or camera index (0 for webcam)
        output_path (str, optional): Path to save output video
        display (bool): Whether to display the processed frames
        depth_interval (int): Process depth every N frames
    """
    # Initialize models
    print("Initializing models...")
    depth_estimator = MiDaSDepthEstimator(model_type="MiDaS_small")  # Use smaller model for real-time performance
    object_detector = YOLOv5Detector(model_name="yolov5s", conf_threshold=0.3)
    fusion = DepthDetectionFusion(depth_estimator, object_detector, depth_processing_interval=depth_interval)
    
    print(f"Depth estimation will run every {depth_interval} frames")
    
    # Open video capture
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        print(f"Opening camera {video_path}...")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Opening video file: {video_path}")
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Process frames
    frame_count = 0
    processing_times = []
    
    print("Processing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
        count_text = f"Frame: {frame_count}"
        cv2.putText(vis_frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display number of detections
        det_text = f"Detections: {len(descriptors)}"
        cv2.putText(vis_frame, det_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame to output video
        if writer:
            writer.write(vis_frame)
        
        # Display frame
        if display:
            cv2.imshow('DepthMOT', vis_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
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

def main():
    parser = argparse.ArgumentParser(description="DepthMOT: 3D Multi-Object Tracking with Depth")
    parser.add_argument("--input", type=str, default="0", help="Path to video file or camera index (default: 0 for webcam)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (optional)")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--depth-interval", type=int, default=3, help="Process depth every N frames (default: 3)")
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, not args.no_display, args.depth_interval)

if __name__ == "__main__":
    main()
