# DepthMOT: 3D Multi-Object Tracking with Depth

This repository contains a proof-of-concept implementation for combining depth estimation (MiDaS) and object detection (YOLOv8) to create 3D descriptors for tracking.

## Overview

The fusion module takes the following approach:
1. Extract x, y coordinates from YOLOv8 bounding boxes
2. Sample depth (z) values from the MiDaS depth map within each bounding box
3. Create a combined 3D descriptor for each detected object
4. Track objects in 3D space using a modified SORT tracker

## Features

- Person detection using YOLOv8
- Depth estimation using MiDaS
- Fusion of detection and depth information into 3D descriptors
- 3D object tracking using a modified SORT algorithm
- Performance optimization by processing depth every N frames
- Visualization of 3D coordinates (x, y, z) for each detected person

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for better performance)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/depthmot.git
   cd depthmot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** The first time you run the script, it will automatically download the YOLOv8 and MiDaS model weights (approximately 100MB each). This may take a few minutes depending on your internet connection.

## Running the Demo

The repository includes two test scripts:

### Basic Fusion Demo

Run the basic fusion demo (without tracking):
```bash
python test_fusion.py
```

### Fusion with Tracking Demo

Run the fusion with SORT tracking:
```bash
python test_fusion_tracking.py
```

These scripts will:
- Use the default video file (`data/viratSample.mp4`)
- Process depth every 5 frames
- Display the processed video with 3D coordinates and tracking IDs

### Customizing the Configuration

You can modify the following settings at the top of the test scripts:

```python
# ======= CONFIGURATION =======
# Change these settings as needed
VIDEO_PATH = "data/viratSample.mp4"  # Path to the video file
DEPTH_INTERVAL = 5                   # Process depth every N frames
SAVE_FRAMES = False                  # Save individual frames as images
OUTPUT_PATH = None                   # Path to save output video (set to None to disable)
DETECT_ONLY_PEOPLE = True            # Only detect people (class 0 in COCO dataset)
# ============================

# SORT Tracker parameters (only in test_fusion_tracking.py)
MAX_AGE = 5                          # Maximum number of frames to keep a track alive without matching
MIN_HITS = 3                         # Minimum number of hits to start a track
DISTANCE_THRESHOLD = 50.0            # Maximum distance for matching detections to tracks
```

### Using Your Own Video

1. Place your video file in the `data` directory
2. Update the `VIDEO_PATH` in the test script to point to your video file

## Technical Details

### 3D Descriptor Format

Each 3D descriptor is a dictionary with the following keys:
- `bbox`: [x1, y1, x2, y2] (coordinates of the 2D bounding box)
- `bbox3d`: [xmin, ymin, zmin, xmax, ymax, zmax] (3D bounding box for SORT)
- `center_3d`: [x, y, z] (3D coordinates of the object center)
- `confidence`: Detection confidence
- `class_id`: Class ID
- `class_name`: Class name
- `depth_stats`: Dictionary with depth statistics within the box
  - `min`: Minimum depth value
  - `max`: Maximum depth value
  - `mean`: Mean depth value
  - `median`: Median depth value (used as the z-coordinate)

### SORT Tracking

The SORT (Simple Online and Realtime Tracking) algorithm has been modified to work with 3D bounding boxes:
- Uses 3D IoU for detection-to-track association
- Kalman filter state includes position, scale, aspect ratio, and depth
- Returns track IDs with 3D center coordinates [x, y, z, track_id]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Understanding the Output

The visualization shows:
- Bounding boxes around detected people
- 3D coordinates (x, y, z) above each detection:
  - x, y: Center coordinates of the bounding box in image space (in pixels)
  - z: Depth value at the center point (normalized to 0-1 range)
- Depth processing indicator (NEW/Cached)
- FPS counter
- Frame counter
- Detection counter

## Performance Considerations

- The script is optimized to run depth estimation every N frames (default: 5) to improve performance
- On an M1 Pro MacBook, the implementation achieves around 7-8 FPS
- Performance will vary based on hardware capabilities
- For real-time applications, consider:
  - Increasing the depth processing interval
  - Using a smaller input resolution
  - Using a lighter depth estimation model

## Project Structure

- `src/depth/`: MiDaS depth estimation implementation
- `src/detection/`: YOLOv8 object detection implementation
- `src/fusion/`: Integration of depth and detection information
- `data/`: Sample videos and test data
- `test_fusion.py`: Demo script for testing the implementation

## Next Steps

This proof-of-concept demonstrates the creation of 3D descriptors by combining depth estimation and object detection. These descriptors can be used for:

1. 3D tracking of objects in a scene
2. Handling occlusions based on depth information
3. Improving identity preservation in tracking
4. Motion prediction in 3D space

## License

[Specify your license here]

## Acknowledgments

- [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [VIRAT Dataset](https://viratdata.org/) for the sample video
