import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
import time

def main(visualize=False):
    # Set benchmark parameters
    test_benchmark = "MOT17"
    test_split = "train"
    tracker_name = "SORT3D-Centroid"
    
    # Get the scripts directory and change to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if TrackEval directory exists
    if not os.path.isdir("TrackEval"):
        print("Error: TrackEval directory not found. Please run the script from the project root.")
        sys.exit(1)
    
    # Create paths for tracker results
    tracker_output_dir = os.path.join("data", "trackers", "mot_challenge", tracker_name, "data")
    os.makedirs(tracker_output_dir, exist_ok=True)
    
    # Check if previous results exist and back them up if needed
    if os.path.exists(tracker_output_dir) and os.listdir(tracker_output_dir):
        backup_time = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = f"{tracker_output_dir}_backup_{backup_time}"
        print(f"Backing up existing results to {backup_dir}")
        shutil.copytree(tracker_output_dir, backup_dir)
    
    # Define training sequences
    sequences = ["MOT17-02-SDP", "MOT17-04-SDP"]
    
    # Create seqmap file
    seqmaps_dir = os.path.join("TrackEval", "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmaps_dir, exist_ok=True)
    
    seqmap_file = os.path.join(seqmaps_dir, f"{test_benchmark}-{test_split}.txt")
    
    # Write sequence map file
    with open(seqmap_file, "w") as f:
        f.write("name\n")
        for seq in sequences:
            f.write(f"{seq}\n")
    
    print(f"Created sequence map file: {seqmap_file}")
    print(f"Processing sequences: {', '.join(sequences)}")
    
    # Process each training sequence with generate_sort3d_results_centroid.py
    for seq in sequences:
        print(f"\nProcessing sequence: {seq}")
        
        expected_output = os.path.join(tracker_output_dir, f"{seq}.txt")
        
        # Command to run centroid-based tracking
        cmd = [
            "python", "generate_sort3d_results_centroid.py",
            "--sequences", seq
        ]
        
        # Add visualization flag if requested
        if visualize:
            cmd.append("--visualize")
        
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(expected_output):
                print(f"Successfully processed {seq}")
                # Verify the output file contains data
                with open(expected_output, 'r') as f:
                    lines = f.readlines()
                    print(f"  Generated {len(lines)} tracking results")
            else:
                print(f"Error: Failed to generate results for {seq}. Output file not found: {expected_output}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {seq}: {e}")
            sys.exit(1)
    
    # Verify all output files exist before running evaluation
    all_files_exist = True
    for seq in sequences:
        output_file = os.path.join(tracker_output_dir, f"{seq}.txt")
        if not os.path.exists(output_file):
            print(f"Error: Output file for {seq} not found at {output_file}")
            all_files_exist = False
    
    if not all_files_exist:
        print("Some output files are missing. Evaluation will not be run.")
        sys.exit(1)
        
    print("\nAll sequences processed successfully")
    
    # Run TrackEval evaluation
    print("\nRunning evaluation...")
    
    eval_cmd = [
        "python", "TrackEval/scripts/run_mot_challenge.py",
        "--BENCHMARK", test_benchmark,
        "--SPLIT_TO_EVAL", test_split,
        "--TRACKERS_TO_EVAL", tracker_name,
        "--METRICS", "HOTA", "CLEAR", "Identity", "Count", "VACE", "JAndF",
        "--USE_PARALLEL", "False"
    ]
    
    try:
        subprocess.run(eval_cmd, check=True)
        print("\nEvaluation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SORT3D-Centroid evaluation on MOT17 dataset")
    parser.add_argument("--visualize", action="store_true", help="Visualize tracking results during generation")
    args = parser.parse_args()
    
    main(visualize=args.visualize) 