import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline SORT algorithm on MOT17 dataset")
    parser.add_argument("--sequences", nargs="+", default=[], help="Specific sequences to evaluate (default: use predefined list)")
    parser.add_argument("--visualize", action="store_true", help="Visualize tracking results")
    args = parser.parse_args()

    # Define benchmark and split parameters
    test_benchmark = "MOT17"
    test_split = "train"

    # Change to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    os.chdir(project_root)
    
    # Check if TrackEval directory exists
    if not os.path.exists("TrackEval"):
        print("Error: TrackEval directory not found. Please make sure you're in the correct directory.")
        sys.exit(1)
    
    # Define absolute paths for better reliability
    trackeval_dir = os.path.abspath("TrackEval")
    
    # Create directories for baseline SORT tracker output with detailed path verification
    tracker_name = "BaselineSORT"
    tracker_dir = os.path.join(trackeval_dir, "data", "trackers", "mot_challenge", f"{test_benchmark}-{test_split}", tracker_name)
    tracker_output_dir = os.path.join(tracker_dir, "data")
    
    print(f"Creating directories for tracker output: {tracker_output_dir}")
    
    # Create each directory level if it doesn't exist
    os.makedirs(os.path.join(trackeval_dir, "data", "trackers"), exist_ok=True)
    os.makedirs(os.path.join(trackeval_dir, "data", "trackers", "mot_challenge"), exist_ok=True)
    os.makedirs(os.path.join(trackeval_dir, "data", "trackers", "mot_challenge", f"{test_benchmark}-{test_split}"), exist_ok=True)
    os.makedirs(tracker_dir, exist_ok=True)
    os.makedirs(tracker_output_dir, exist_ok=True)
    
    # Verify that directories were created
    print(f"Tracker directory exists: {os.path.exists(tracker_dir)}")
    print(f"Tracker output directory exists: {os.path.exists(tracker_output_dir)}")
    
    # Define all MOT17 train sequences or use the provided ones
    if args.sequences:
        train_sequences = args.sequences
    else:
        # MOT17 train consists of 2 sequences, each with 3 detector versions (DPM, FRCNN, SDP)
        train_sequences = [
            "MOT17-02-SDP",
            "MOT17-04-SDP"
        ]
    
    # Define absolute paths for better reliability
    seqmaps_dir = os.path.join(trackeval_dir, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmaps_dir, exist_ok=True)
    
    # Create a sequence map file with the default expected name
    default_seqmap_name = f"{test_benchmark}-{test_split}.txt" # e.g., MOT17-train.txt
    test_seqmap_path = os.path.join(seqmaps_dir, default_seqmap_name)
    
    # Make sure the seqmap file has the proper format (header + sequences)
    with open(test_seqmap_path, 'w') as f:
        f.write("name\n")  # Write the header line
        for seq in train_sequences:
            f.write(seq + "\n") # Write each sequence name on a new line
    
    print(f"Created sequence map file at {test_seqmap_path}")
    print(f"File exists: {os.path.isfile(test_seqmap_path)}")
    
    # Path to generate_baseline_sort_results.py in src/tests
    generator_script = os.path.join("src", "tests", "generate_baseline_sort_results.py")
    if not os.path.exists(generator_script):
        print(f"Error: {generator_script} not found.")
        sys.exit(1)
    
    # Run baseline SORT tracker on all sequences
    print(f"Running baseline SORT tracker on all MOT17 train sequences...")
    for sequence in train_sequences:
        print(f"Processing {sequence}...")
        try:
            cmd = ["python", generator_script, "--sequences", sequence, "--output_dir", tracker_output_dir]
            if args.visualize:
                cmd.append("--visualize")
            
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {sequence}")
        except subprocess.CalledProcessError as e:
            print(f"Error running baseline SORT tracker on {sequence}: {e}")
    
    print("Baseline SORT tracker completed for all sequences.")
    
    # Run TrackEval evaluation
    print("Running TrackEval evaluation...")
    os.chdir("TrackEval")
    try:
        command = [
            "python", "scripts/run_mot_challenge.py",
            "--BENCHMARK", "MOT17",
            "--SPLIT_TO_EVAL", "train",
            "--TRACKERS_TO_EVAL", tracker_name,
            "--METRICS", "HOTA", "CLEAR", "Identity", "Count", "VACE", "JAndF",
            "--USE_PARALLEL", "False"
        ]
        
        subprocess.run(command, check=True)
        print("TrackEval evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running TrackEval evaluation: {e}")

    os.chdir(project_root)
    
    print("All processing completed successfully!")

if __name__ == "__main__":
    main() 