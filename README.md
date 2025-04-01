# DepthMOT: 3D Multi-Object Tracking with Depth

This repository contains implementations of multi-object tracking algorithms that utilize depth information for improved tracking performance.

## Key Tracking Methods

### SortIOU Tracker
- An extension of the SORT algorithm that uses IOU (Intersection over Union) for association
- Works with 6D bounding boxes including depth information [x1, y1, z1, x2, y2, z2]
- Provides more robust tracking by considering object depth in the matching process
- Found in `src/tracking/sortIOU.py`

### SortEuclidean Tracker
- Uses centroids and Euclidean distance for track association
- Tracks objects using 3D points [x, y, z] where z represents depth
- More suitable for scenarios where object size changes rapidly
- Efficient for tracking when only center points matter
- Found in `src/tracking/sortEuclidean.py`

## Overview

The system takes the following approach:
1. Extract x, y coordinates from YOLOv8 bounding boxes
2. Sample depth (z) values from the MiDaS depth map within each bounding box
3. Create a combined 3D descriptor for each detected object
4. Track objects in 3D space using one of our tracking methods

## Features

- Person detection using YOLOv8
- Depth estimation using MiDaS
- Fusion of detection and depth information into 3D descriptors
- 3D object tracking with tracking ID visualization
- Performance optimization by processing depth every N frames

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

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** The first time you run the script, it will automatically download the YOLOv8 and MiDaS model weights.

## Project Structure

- `src/tracking/`: Tracking implementation (SortIOU and SortEuclidean)
- `src/depth/`: MiDaS depth estimation implementation
- `src/detection/`: YOLOv8 object detection implementation
- `src/fusion/`: Integration of depth and detection information
- `src/tests/`: Test scripts and visualization tools

## Tracking Implementation Details

The tracking implementations are built on the SORT (Simple Online and Realtime Tracking) framework:

- **SortIOU**: Uses 3D IoU for detection-to-track association with Kalman filtering for motion prediction.
- **SortEuclidean**: Uses Euclidean distance between 3D centroids for track association, which can be more effective for objects that change size rapidly.

Both trackers handle:
- Track creation and deletion based on detection confidence
- Occlusions using depth information
- Identity preservation across frames
