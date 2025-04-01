import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Use absolute imports from project root
from src.detection.yolo_detector import YOLOv5Detector
from src.tracking.baselineSort import Sort

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
    Process a MOT17 sequence through baseline SORT tracker
    
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
    detector = YOLOv5Detector(model_name="yolov8s", conf_threshold=0.35)
    tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.25)
    
    # Output file path
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.basename(seq_path)}.txt")
    
    # Initialize visualization
    vis_window = None
    if visualize:
        vis_window = f"Tracking: {seq_name}"
        cv2.namedWindow(vis_window, cv2.WINDOW_NORMAL)
    
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
        
        # Get detections from YOLOv8
        detections = detector.detect(frame)
        
        # Convert detections to SORT format
        if detections:
            sort_dets = np.zeros((len(detections), 5))
            for i, det in enumerate(detections):
                # Get bounding box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = det['bbox']
                # Append bounding box with confidence score [x1, y1, x2, y2, score]
                sort_dets[i, :] = [x1, y1, x2, y2, det['confidence']]
        else:
            sort_dets = np.empty((0, 5))
        
        # Update SORT tracker (returns tracks with [x1, y1, x2, y2, id])
        tracks = tracker.update(sort_dets)
        
        # Store results in MOT Challenge format
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            w, h = x2 - x1, y2 - y1
            results.append(f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        
        if visualize:
            vis_frame = frame.copy()
            
            # Draw detections (green)
            for i in range(sort_dets.shape[0]):
                x1, y1, x2, y2, score = sort_dets[i, :]
                cv2.rectangle(vis_frame, 
                              (int(x1), int(y1)), 
                              (int(x2), int(y2)), 
                              (0, 255, 0), 2)
            
            # Draw tracks (blue with ID)
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                
                # Draw tracked bounding box
                cv2.rectangle(vis_frame, 
                             (int(x1), int(y1)), 
                             (int(x2), int(y2)), 
                             (255, 0, 0), 2)
                
                # Draw track ID
                cv2.putText(vis_frame, 
                           f"ID:{int(track_id)}", 
                           (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow(vis_window, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
    parser = argparse.ArgumentParser(description="Generate baseline SORT results for MOT17 dataset")
    
    parser.add_argument("--data_dir", type=str, default="data/MOT17/MOT17/train", 
                        help="Path to MOT17 dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/trackers/mot_challenge/MOT17-train/BaselineSORT/data", 
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