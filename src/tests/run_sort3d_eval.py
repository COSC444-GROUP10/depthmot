import os
import sys
import subprocess
import shutil

def main():
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
    
    # Create directories for SORT3D tracker output
    tracker_output_dir = os.path.join("data", "trackers", "mot_challenge", "MOT17-train", "SORT3D", "data")
    os.makedirs(tracker_output_dir, exist_ok=True)
    
    train_sequences = [
        "MOT17-02-SDP",
        "MOT17-04-SDP"
    ]
    
    # Define absolute paths for better reliability
    trackeval_dir = os.path.abspath("TrackEval")
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
    
    # Path to generate_sort3d_results.py in src/tests
    generator_script = os.path.join("src", "tests", "generate_sort3d_results.py")
    
    # Run SORT3D tracker on all sequences
    print(f"Running SORT3D tracker on all MOT17 train sequences...")
    for sequence in train_sequences:
        print(f"Processing {sequence}...")
        try:
            subprocess.run([
                "python", generator_script,
                "--sequences", sequence
            ], check=True)
            print(f"Successfully processed {sequence}")
        except subprocess.CalledProcessError as e:
            print(f"Error running SORT3D tracker on {sequence}: {e}")
            # Continue with other sequences even if one fails
    
    print("SORT3D tracker completed for all sequences.")
    
    # Run TrackEval evaluation
    print("Running TrackEval evaluation...")
    os.chdir("TrackEval")
    try:
        # Update this to include only SORT3D
        command = [
            "python", "scripts/run_mot_challenge.py",
            "--BENCHMARK", "MOT17",
            "--SPLIT_TO_EVAL", "train",
            "--TRACKERS_TO_EVAL", "SORT3D", 
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