import numpy as np
import os
import cv2
import time
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Use absolute imports from project root
from src.depth.midas_depth import MiDaSDepthEstimator
from src.detection.yolo_detector import YOLOv5Detector
from src.fusion.descriptor_fusion import DepthDetectionFusion
from src.tracking.sortEuclidean import SortEuclidean

def load_sequence_info(seq_path):
    """Load sequence information from seqinfo.ini file"""
    seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
    info = {}
    
    with open(seqinfo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('['):
                key, value = line.split('=')
                info[key.strip()] = value.strip()
    
    return info

def process_sequence(seq_path, output_dir, visualize=False):
    """
    Process a MOT17 sequence through centroid-based SORT tracker
    
    Args:
        seq_path: Path to sequence directory
        output_dir: Directory to save tracking results
        visualize: Whether to visualize tracking results
    """
    print(f"Processing sequence: {os.path.basename(seq_path)}")
    
    # Load sequence info
    seq_info = load_sequence_info(seq_path)
    seq_name = seq_info.get('name', os.path.basename(seq_path))
    seq_length = int(seq_info.get('seqLength', 0))
    img_width = int(seq_info.get('imWidth', 0))
    img_height = int(seq_info.get('imHeight', 0))
    
    # Setup image directory
    img_dir = os.path.join(seq_path, seq_info.get('imDir', 'img1'))
    
    # Initialize models
    print("Initializing models...")
    
    # Initialize depth estimator
    depth_model = "DPT_Large" #DPT_Large
    depth_estimator = MiDaSDepthEstimator(model_type=depth_model)
    
    # Initialize object detector
    detector_model = "yolov8s"
    detector_confidence = 0.35
    detector = YOLOv5Detector(model_name=detector_model, conf_threshold=detector_confidence)
    
    # Initialize fusion module
    fusion = DepthDetectionFusion(depth_estimator, detector, depth_processing_interval=3)
    
    # Initialize tracker
    max_age = 50
    min_hits = 3
    distance_threshold = 50.0
    tracker = SortEuclidean(max_age=max_age, min_hits=min_hits, distance_threshold=distance_threshold)
    tracker_name = "SORT3D-Centroid"
    
    # Create output directory
    tracker_output_dir = os.path.join(output_dir, tracker_name, "data")
    os.makedirs(tracker_output_dir, exist_ok=True)
    output_file = os.path.join(tracker_output_dir, f"{os.path.basename(seq_path)}.txt")
    
    # Initialize visualization
    vis_window = None
    if visualize:
        vis_window = f"Centroid Tracking: {seq_name}"
        cv2.namedWindow(vis_window, cv2.WINDOW_NORMAL)
    
    # Initialize storage for detection sizes for better bbox estimation
    detection_sizes = []
    
    # Process each frame
    results = []
    for frame_id in tqdm(range(1, seq_length + 1), desc=f"Processing {seq_name}"):
        # Load image
        img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} does not exist!")
            continue
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Failed to load image {img_path}!")
            continue
        
        # Process frame using the full pipeline (depth + detection)
        descriptors = fusion.process_frame(frame)
        
        # Update detection sizes for better bbox estimation
        if descriptors:
            for desc in descriptors:
                x1, y1, x2, y2 = desc['bbox']
                w, h = x2 - x1, y2 - y1
                detection_sizes.append((w, h))
                # Keep only the most recent 100 detections for size estimation
                if len(detection_sizes) > 100:
                    detection_sizes.pop(0)
        
        # Convert descriptors to SORT format and update tracker
        if descriptors:
            detections = fusion.get_sort_detections(descriptors)
            tracks = tracker.update(detections)
        else:
            tracks = tracker.update(np.empty((0, 7)))
        
        # Calculate average detection size for bbox estimation
        avg_width, avg_height = 50, 100  # Default values
        if detection_sizes:
            avg_width = sum(w for w, h in detection_sizes) / len(detection_sizes)
            avg_height = sum(h for w, h in detection_sizes) / len(detection_sizes)
        
        # Store results in MOT Challenge format
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        for track in tracks:
            # Unpack centroid and ID
            center_x, center_y, center_z, track_id = track
            
            # Calculate bounding box from centroid
            x1 = center_x - avg_width / 2
            y1 = center_y - avg_height / 2
            w, h = avg_width, avg_height
            
            # Format in MOT Challenge format
            results.append(f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        
        
        if visualize:
            vis_frame = frame.copy()
            
            # Draw detections if available
            if descriptors:
                for desc in descriptors:
                    x1, y1, x2, y2 = [int(coord) for coord in desc['bbox']]
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw tracks
            if tracks.size > 0:
                for track in tracks:
                    center_x, center_y, center_z, track_id = track
                    
                    # Calculate bounding box from centroid
                    x1 = int(center_x - avg_width / 2)
                    y1 = int(center_y - avg_height / 2)
                    x2 = int(center_x + avg_width / 2)
                    y2 = int(center_y + avg_height / 2)
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Draw track ID and depth
                    label = f"ID:{int(track_id)} Z:{center_z:.2f}m"
                    cv2.putText(vis_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw centroid
                    cv2.circle(vis_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
            
            # Add tracker info
            cv2.putText(vis_frame, f"Tracker: {tracker_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_frame, f"Frame: {frame_id}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis_frame, f"Tracks: {len(tracks) if tracks is not None else 0}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
            cv2.imshow(vis_window, vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.writelines(results)
    
    print(f"Results saved to {output_file}")
    
    # Clean up visualization
    if visualize:
        cv2.destroyWindow(vis_window)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate Centroid-Based SORT3D results for MOT17 dataset")
    
    parser.add_argument("--data_dir", type=str, default="data/MOT17/MOT17/train", 
                        help="Path to MOT17 dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/trackers/mot_challenge", 
                        help="Output directory to save tracking results")
    parser.add_argument("--visualize", action="store_true", help="Visualize tracking results")
    parser.add_argument("--sequences", type=str, nargs="+", default=[], 
                        help="Specific sequences to process (default: all)")
    
    args = parser.parse_args()
    
    # Get list of sequences
    if args.sequences:
        seq_paths = [os.path.join(args.data_dir, seq) for seq in args.sequences]
    else:
        seq_paths = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                    if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # Process each sequence
    for seq_path in seq_paths:
        process_sequence(seq_path, args.output_dir, args.visualize)
    
    print("All sequences processed successfully!")

if __name__ == "__main__":
    main() 