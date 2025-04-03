"""
    SORT: A Simple, Online and Realtime Tracker (Modified for 3D using Euclidean distance)
    Original Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
"""

from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)

def euclidean_distance_batch(centroids_test, centroids_gt):
    """
    Compute Euclidean distance between centroids
    
    Input:
    - centroids_test: [N, 3] test centroids [x, y, z]
    - centroids_gt: [M, 3] ground truth centroids [x, y, z]
    
    Output:
    - dist: [N, M] distance matrix
    """
    N = centroids_test.shape[0]
    M = centroids_gt.shape[0]
    
    test_exp = np.expand_dims(centroids_test, 1)  # [N, 1, 3]
    gt_exp = np.expand_dims(centroids_gt, 0)      # [1, M, 3]
    
    diff = test_exp - gt_exp  # [N, M, 3]
    dist_sq = np.sum(diff ** 2, axis=2)  # [N, M]
    
    return np.sqrt(dist_sq)

def extract_centroid(det):
    """
    Extract centroid from detection
    
    Input:
    - det: [xmin, ymin, zmin, xmax, ymax, zmax, score]
    
    Output:
    - centroid_score: [x, y, z, score]
    """
    xmin, ymin, zmin, xmax, ymax, zmax, score = det
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    z = (zmin + zmax) / 2.0  # Use existing z-average
    return np.array([x, y, z, score])

class KalmanBoxTracker:
    """
    Kalman filter-based tracker for 3D centroids
    State vector: [x, y, z, vx, vy, vz]
    """
    count = 0
    
    def __init__(self, centroid_score):
        """
        Initialize tracker with centroid and score
        
        Input:
        - centroid_score: [x, y, z, score]
        """
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x, vx
            [0, 1, 0, 0, 1, 0],  # y, vy
            [0, 0, 1, 0, 0, 1],  # z, vz
            [0, 0, 0, 1, 0, 0],  # vx
            [0, 0, 0, 0, 1, 0],  # vy
            [0, 0, 0, 0, 0, 1]   # vz
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0]   # z
        ])
        
        self.kf.R *= 10.0
        self.kf.P[3:, 3:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[3:, 3:] *= 0.01
        
        self.kf.x[:3] = centroid_score[:3].reshape((3, 1))
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, centroid_score):
        """
        Update tracker with new centroid
        
        Input:
        - centroid_score: [x, y, z, score]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(centroid_score[:3].reshape((3, 1)))
    
    def predict(self):
        """
        Predict next centroid position
        
        Output:
        - centroid: [x, y, z]
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:3].flatten())
        return self.history[-1]
    
    def get_state(self):
        """
        Get current centroid state
        
        Output:
        - centroid: [x, y, z]
        """
        return self.kf.x[:3].flatten()

def associate_detections_to_trackers(detections, trackers, distance_threshold=50.0):
    """
    Associate detections to trackers using Euclidean distance
    
    Input:
    - detections: [N, 4] centroids [x, y, z, score]
    - trackers: [M, 3] predicted centroids [x, y, z]
    - distance_threshold: max distance for a valid match
    
    Output:
    - matches: [K, 2] matched indices (detection_idx, tracker_idx)
    - unmatched_detections: [L] unmatched detection indices
    - unmatched_trackers: [P] unmatched tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    dist_matrix = euclidean_distance_batch(detections[:, :3], trackers)
    
    if min(dist_matrix.shape) > 0:
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
    else:
        matched_indices = np.empty((0, 2), dtype=int)
    
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))
    
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] <= distance_threshold:
            matches.append(m.reshape(1, 2))
            unmatched_detections.remove(m[0])
            unmatched_trackers.remove(m[1])
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class SortEuclidean:
    """
    SORT with 3D centroids and Euclidean distance
    """
    def __init__(self, max_age=1, min_hits=3, distance_threshold=50.0):
        """
        Initialize SORT tracker
        
        Args:
            max_age: Max frames to keep unmatched tracks
            min_hits: Min hits to output a track
            distance_threshold: Max centroid distance for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets=np.empty((0, 7))):
        """
        Update tracker with new detections
        
        Args:
            dets: [N, 7] detections [xmin, ymin, zmin, xmax, ymax, zmax, score]
            
        Returns:
            [M, 4] tracked centroids [x, y, z, id]
        """
        self.frame_count += 1
        
        # Convert detections to centroids
        if dets.size > 0:
            dets_centroids = np.array([extract_centroid(det) for det in dets])
        else:
            dets_centroids = np.empty((0, 4))
        
        # Predict existing trackers
        trks = np.zeros((len(self.trackers), 3))
        to_del = []
        ret = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_centroids, trks, self.distance_threshold)
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets_centroids[m[0]])
        
        # Initialize new trackers
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets_centroids[i])
            self.trackers.append(trk)
        
        # Collect output
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 4))

# Example usage
if __name__ == "__main__":
    seq_dets = np.array([
        [10.0, 20.0, 30.0, 20.0, 40.0, 40.0, 0.9],
        [100.0, 120.0, 30.0, 120.0, 140.0, 40.0, 0.8]
    ])
    
    mot_tracker = SortEuclidean(max_age=1, min_hits=1, distance_threshold=50.0)
    
    trackers = mot_tracker.update(seq_dets)
    
    for d in trackers:
        print(f"ID: {int(d[3])}, Centroid: ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})")