"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
# Remove matplotlib dependency
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
    Compute 3D IoU between bounding boxes
    
    Input:
    - bb_test: [N, 6] test bounding boxes
    - bb_gt: [M, 6] ground truth bounding boxes
    
    Output:
    - iou: [N, M] IoU matrix
    """
    N = bb_test.shape[0]
    M = bb_gt.shape[0]
    
    # Expand dimensions to compute IoU for all pairs
    bb_test = np.expand_dims(bb_test, 1)  # [N, 1, 6]
    bb_gt = np.expand_dims(bb_gt, 0)      # [1, M, 6]
    
    # Compute intersection
    xx1 = np.maximum(bb_test[:, :, 0], bb_gt[:, :, 0])
    yy1 = np.maximum(bb_test[:, :, 1], bb_gt[:, :, 1])
    zz1 = np.maximum(bb_test[:, :, 2], bb_gt[:, :, 2])
    xx2 = np.minimum(bb_test[:, :, 3], bb_gt[:, :, 3])
    yy2 = np.minimum(bb_test[:, :, 4], bb_gt[:, :, 4])
    zz2 = np.minimum(bb_test[:, :, 5], bb_gt[:, :, 5])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    d = np.maximum(0., zz2 - zz1)
    
    intersection = w * h * d
    
    # Compute volumes
    vol_test = (bb_test[:, :, 3] - bb_test[:, :, 0]) * \
               (bb_test[:, :, 4] - bb_test[:, :, 1]) * \
               (bb_test[:, :, 5] - bb_test[:, :, 2])
    vol_gt = (bb_gt[:, :, 3] - bb_gt[:, :, 0]) * \
             (bb_gt[:, :, 4] - bb_gt[:, :, 1]) * \
             (bb_gt[:, :, 5] - bb_gt[:, :, 2])
    
    union = vol_test + vol_gt - intersection
    
    # Compute IoU
    iou = intersection / union
    
    # Handle division by zero
    iou = np.where(union > 0, iou, 0)
    
    return iou

def convert_bbox_to_z(bbox):
    """
    Convert 6D bounding box to state vector [x, y, z, s, r, d]
    where:
    - x, y, z are the center coordinates
    - s is the scale (average of width and height)
    - r is the aspect ratio (width / height)
    - d is the depth range (zmax - zmin)
    
    Input:
    - bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
    
    Output:
    - z: [x, y, z, s, r, d]
    """
    w = bbox[3] - bbox[0]
    h = bbox[4] - bbox[1]
    d = bbox[5] - bbox[2]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    z = bbox[2] + d/2.
    s = (w + h) / 2  # scale is average of width and height
    r = w / float(h) if h > 0 else 1.0  # aspect ratio
    return np.array([x, y, z, s, r, d]).reshape((6, 1))


def convert_x_to_bbox(x,score=None):
    """
    Convert state vector [x, y, z, s, r, d, vx, vy, vz, vs, vd] to 6D bounding box [xmin, ymin, zmin, xmax, ymax, zmax]
    
    Input:
    - x: state vector [x, y, z, s, r, d, vx, vy, vz, vs, vd]
    
    Output:
    - bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    center_x = float(x[0].item())
    center_y = float(x[1].item())
    center_z = float(x[2].item())
    scale = float(x[3].item())
    ratio = float(x[4].item())
    depth = float(x[5].item())
    
    width = scale * ratio
    height = scale / ratio
    
    xmin = center_x - width/2
    ymin = center_y - height/2
    zmin = center_z - depth/2
    xmax = center_x + width/2
    ymax = center_y + height/2
    zmax = center_z + depth/2
    if(score==None):
        return np.array([xmin, ymin, zmin, xmax, ymax, zmax]).flatten()
    else:
        return np.array([xmin, ymin, zmin, xmax, ymax, zmax, score]).flatten()

class KalmanBoxTracker(object):
    """
    Kalman filter-based tracker for 6D bounding boxes
    State vector: [x, y, z, s, r, d, vx, vy, vz, vs, vd]
    where:
    - x, y, z are the center coordinates
    - s is the scale (average of width and height)
    - r is the aspect ratio (width / height)
    - d is the depth range (zmax - zmin)
    - vx, vy, vz, vs, vd are the respective velocities
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize a tracker with a 6D bounding box and score
        
        Input:
        - bbox: [xmin, ymin, zmin, xmax, ymax, zmax, score]
        """
        # Define Kalman filter with 11 state variables and 6 measurement variables
        self.kf = KalmanFilter(dim_x=11, dim_z=6)
        
        # Format: [x, y, z, s, r, d, vx, vy, vz, vs, vd]
        self.kf.F = np.array([
            # Position, scale, ratio, depth              Velocity components
            [1, 0, 0, 0, 0, 0,    1, 0, 0, 0, 0],  # x  vx
            [0, 1, 0, 0, 0, 0,    0, 1, 0, 0, 0],  # y  vy
            [0, 0, 1, 0, 0, 0,    0, 0, 1, 0, 0],  # z  vz
            [0, 0, 0, 1, 0, 0,    0, 0, 0, 1, 0],  # s  vs
            [0, 0, 0, 0, 1, 0,    0, 0, 0, 0, 0],  # r 
            [0, 0, 0, 0, 0, 1,    0, 0, 0, 0, 1],  # d  vd
            [0, 0, 0, 0, 0, 0,    1, 0, 0, 0, 0],  # vx
            [0, 0, 0, 0, 0, 0,    0, 1, 0, 0, 0],  # vy
            [0, 0, 0, 0, 0, 0,    0, 0, 1, 0, 0],  # vz
            [0, 0, 0, 0, 0, 0,    0, 0, 0, 1, 0],  # vs
            [0, 0, 0, 0, 0, 0,    0, 0, 0, 0, 1]   # vd
        ])
        
        # Define measurement matrix H explicitly (we only observe position, scale, ratio, and depth)
        # This maps the state vector to the measurement vector
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # z
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # s
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # r
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]   # d
        ])
        
        self.kf.R[3:,3:] *= 10.0   
        self.kf.P[6:, 6:] *= 1000.0  # High uncertainty in velocity components
        self.kf.P *= 10.0            
        self.kf.Q[9, 9] *= 0.01 
        self.kf.Q[6:, 6:] *= 0.01 

        self.kf.x[0:6] = convert_bbox_to_z(bbox[0:6])
        
        # Tracking metadata
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """
        Update the tracker with a new detection
        
        Input:
        - bbox: [xmin, ymin, zmin, xmax, ymax, zmax, score]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox[0:6]))
        
    
    def predict(self):
        """
        Predict the next state and return the predicted bounding box
        
        Output:
        - bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
        """

        if((self.kf.x[9]+self.kf.x[3])<=0):  # If scale + scale_velocity <= 0
            self.kf.x[9] *= 0.0               # Reset scale velocity

        if((self.kf.x[10]+self.kf.x[5])<=0):  # If depth + depth_velocity <= 0
            self.kf.x[10] *= 0.0    
        
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """
        Get the current state as a bounding box
        
        Output:
        - bbox: [xmin, ymin, zmin, xmax, ymax, zmax, score]
        """
        return convert_x_to_bbox(self.kf.x)

    
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Associate detections to trackers using IoU
    
    Input:
    - detections: [N, 7] detection bounding boxes with scores [xmin, ymin, zmin, xmax, ymax, zmax, score]
    - trackers: [M, 6] tracker bounding boxes [xmin, ymin, zmin, xmax, ymax, zmax]
    - iou_threshold: minimum IoU for a valid match
    
    Output:
    - matches: [K, 2] matched indices (detection_idx, tracker_idx)
    - unmatched_detections: [L] unmatched detection indices
    - unmatched_trackers: [P] unmatched tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 7), dtype=int)
    
    # Compute IoU between detections and trackers (using only the bbox part, not score)
    iou_matrix = iou_batch(detections[:, 0:6], trackers)
    
    # Convert IoU to cost matrix (higher IoU = lower cost)
    cost_matrix = 1 - iou_matrix
    
    # Use Hungarian algorithm to find optimal assignment
    if min(cost_matrix.shape) > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
    else:
        matched_indices = np.empty((0, 2), dtype=int)
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort:
    """
    SORT: Simple Online and Realtime Tracking with 6D bounding boxes
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker
        
        Args:
            max_age: Maximum number of frames to keep a track alive without matching
            min_hits: Minimum number of hits to start a track
            iou_threshold: Minimum IoU for a valid match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets=np.empty((0, 7))):
        """
        Update the tracker with new detections
        
        Args:
            dets: [N, 7] array of detections in format [xmin, ymin, zmin, xmax, ymax, zmax, score]
            
        Returns:
            A similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Filter out invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0],:])
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,7))

# Example usage
if __name__ == "__main__":
    # Simulated detections: [xmin, ymin, zmin, xmax, ymax, zmax, score]
    seq_dets = np.array([
        [10.0, 20.0, 30.0, 20.0, 40.0, 40.0, 0.9],
        [100.0, 120.0, 30.0, 120.0, 140.0, 40.0, 0.8],
        [11.0, 21.0, 31.0, 21.0, 41.0, 41.0, 0.7],
        [101.0, 121.0, 31.0, 121.0, 141.0, 41.0, 0.6]
    ])
    
    mot_tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
    
    # Process detections
    trackers = mot_tracker.update(seq_dets)
    
    # Print tracking results
    for d in trackers:
        print(f"ID: {int(d[6])}, Bbox: ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}, {d[3]:.2f}, {d[4]:.2f}, {d[5]:.2f})")

