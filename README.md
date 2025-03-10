# depthmot
Improving Multi-Object Tracking Using Depth - A Comprehensive Literature Review

## Overview
This repository contains a literature review conducted by Group 10 for COSC 444/544: Computer Vision. The project explores Multi-Object Tracking (MOT) challenges, focusing on occlusion and ID switching in pedestrian tracking. We evaluate traditional methods like the Kalman Filter (e.g., SORT) and propose integrating depth information to enhance tracking accuracy and robustness.

### Authors
- Allan Cheboiwo 
- Tarek Alkabbani 
- Haoyu Wang 
- Vanessa Laurel Hariyanto

## Abstract
Multiple Object Tracking (MOT) is a critical computer vision task used in surveillance, autonomous driving, and traffic monitoring. Despite advances in deep learning, challenges like occlusion and ID switching persist, especially in pedestrian tracking. This review analyzes traditional methods (e.g., Kalman Filter, Hungarian Algorithm) and proposes a depth-enhanced approach to mitigate these issues, balancing accuracy and efficiency for real-world applications.

## Contents
- **[Literature Review Report](docs/Literature%20Review%20Report.pdf)**: Full PDF of our comprehensive review, including related works, proposed method, and future directions.
- **References**: Cited works are listed in the report.

## Proposed Method
We suggest extending the SORT framework by incorporating depth data (e.g., from LIDAR or depth estimation models) to improve object association and reduce occlusion-related errors. Key components:
- **Object Detection**: Extract bounding boxes and descriptors.
- **Depth Estimation**: Integrate depth maps into tracking.
- **Kalman Filter + Hungarian Algorithm**: Predict and match states with depth-aware data.
- **SIFT (Optional)**: Enhance re-identification for unmatched objects.

See Section 4 of the report for technical details and algorithm diagram.

## Getting Started
This is primarily a literature review repository. To view the report:
1. Clone the repo:
   ```bash
   git clone https://github.com/COSC444-GROUP10/depthmot.git
