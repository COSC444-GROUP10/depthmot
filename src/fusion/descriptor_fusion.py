import numpy as np
import cv2

class DepthDetectionFusion:
    def __init__(self, depth_estimator, object_detector, depth_processing_interval=1):
        """
        Initialize the fusion module that combines depth and detection information
        
        Args:
            depth_estimator: MiDaS depth estimator instance
            object_detector: YOLOv5 detector instance
            depth_processing_interval: Process depth every N frames (default: 1)
        """
        self.depth_estimator = depth_estimator
        self.object_detector = object_detector
        self.depth_processing_interval = depth_processing_interval
        self.frame_count = 0
        self.last_depth_map = None
    
    def process_frame(self, frame):
        """
        Process a frame to get combined 3D descriptors
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            list: List of 3D descriptors, each descriptor is a dictionary with keys:
                - 'bbox': [x1, y1, x2, y2] (coordinates of the bounding box)
                - 'bbox3d': [xmin, ymin, zmin, xmax, ymax, zmax] (3D bounding box for SORT)
                - 'center_3d': [x, y, z] (3D coordinates of the object center)
                - 'confidence': Detection confidence
                - 'class_id': Class ID
                - 'class_name': Class name
                - 'depth_stats': {'min', 'max', 'mean', 'median'} (depth statistics within the box)
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Get depth map (only on certain frames)
        if self.last_depth_map is None or self.frame_count % self.depth_processing_interval == 0:
            self.last_depth_map = self.depth_estimator.estimate_depth(frame)
            is_new_depth = True
        else:
            is_new_depth = False
        
        # Get object detections (on every frame)
        detections = self.object_detector.detect(frame)
        
        # Create 3D descriptors
        descriptors = []
        
        for detection in detections:
            # Extract bounding box
            x1, y1, x2, y2 = detection['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image boundaries
            height, width = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)
            
            # Calculate center point in image coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Extract depth values within the bounding box
            depth_roi = self.last_depth_map[y1:y2, x1:x2]
            
            # Calculate depth statistics
            if depth_roi.size > 0:
                depth_min = float(np.min(depth_roi))
                depth_max = float(np.max(depth_roi))
                depth_mean = float(np.mean(depth_roi))
                depth_median = float(np.median(depth_roi))
                
                # Use median depth as the z-coordinate
                center_z = depth_median
            else:
                # Fallback if ROI is empty
                depth_min = depth_max = depth_mean = depth_median = 0.0
                center_z = 0.0
            
            # Create 3D bounding box for SORT tracker [xmin, ymin, zmin, xmax, ymax, zmax]
            bbox3d = [float(x1), float(y1), depth_min, float(x2), float(y2), depth_max]
            
            # Create 3D descriptor
            descriptor = {
                'bbox': detection['bbox'],
                'bbox3d': bbox3d,
                'center_3d': [float(center_x), float(center_y), float(center_z)],
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'depth_stats': {
                    'min': depth_min,
                    'max': depth_max,
                    'mean': depth_mean,
                    'median': depth_median
                },
                'is_new_depth': is_new_depth
            }
            
            descriptors.append(descriptor)
        
        return descriptors
    
    def get_sort_detections(self, descriptors):
        """
        Convert descriptors to format suitable for SORT tracker
        
        Args:
            descriptors (list): List of 3D descriptors from process_frame
            
        Returns:
            numpy.ndarray: Array of shape (n, 7) with each row containing [xmin, ymin, zmin, xmax, ymax, zmax, score]
                           where score is the detection confidence
        """
        if not descriptors:
            return np.empty((0, 7))
        
        # Extract 3D bounding boxes and confidence scores for SORT
        sort_dets = np.zeros((len(descriptors), 7))
        for i, desc in enumerate(descriptors):
            # Copy bounding box coordinates
            sort_dets[i, 0:6] = desc['bbox3d']
            # Add confidence score
            sort_dets[i, 6] = desc['confidence']
            
        return sort_dets
    

    
    def visualize(self, frame, descriptors):
        """
        Visualize the 3D descriptors on the frame
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            descriptors (list): List of 3D descriptors
            
        Returns:
            numpy.ndarray: Visualization frame
        """
        vis_frame = frame.copy()
        
        # Add depth processing indicator
        if descriptors and descriptors[0].get('is_new_depth', False):
            cv2.putText(vis_frame, "Depth: NEW", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "Depth: Cached", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        for desc in descriptors:
            # Extract information
            x1, y1, x2, y2 = [int(coord) for coord in desc['bbox']]
            center_x, center_y, center_z = desc['center_3d']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # Create label with 3D coordinates
            coords_label = f"X:{center_x:.1f} Y:{center_y:.1f} Z:{center_z:.3f}"
            
            # Draw coordinates label
            cv2.putText(vis_frame, coords_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_frame
    
    def visualize_with_tracking(self, frame, descriptors, tracks):
        """
        Visualize the tracking information on the frame (only one box with ID per tracked object)
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            descriptors (list): List of 3D descriptors
            tracks (numpy.ndarray): Tracking results from SORT tracker with shape (n, 7)
                                   where each row is [xmin, ymin, zmin, xmax, ymax, zmax, track_id]
            
        Returns:
            numpy.ndarray: Visualization frame with tracking information
        """
        # Make a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Add depth processing indicator
        if descriptors and descriptors[0].get('is_new_depth', False):
            cv2.putText(vis_frame, "Depth: NEW", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "Depth: Cached", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Add tracking information
        if tracks is not None and tracks.shape[0] > 0:
            for track in tracks:
                xmin, ymin, zmin, xmax, ymax, zmax, track_id = track
                # Calculate center for visualization
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                
                # Draw tracked bounding box
                cv2.rectangle(vis_frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                
                # Draw track ID and depth range
                cv2.putText(vis_frame, f"ID:{int(track_id)}", 
                           (int(xmin), int(ymin) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw track center point
                cv2.circle(vis_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        
        return vis_frame
    
    def visualize_3d(self, frame, descriptors, tracks=None):
        """
        Visualize with 3D bounding boxes
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            descriptors (list): List of 3D descriptors
            tracks (numpy.ndarray, optional): Tracking results from SORT tracker
            
        Returns:
            numpy.ndarray: Visualization frame with 3D boxes
        """
        # Make a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Add depth processing indicator
        if descriptors and descriptors[0].get('is_new_depth', False):
            cv2.putText(vis_frame, "Depth: NEW", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "Depth: Cached", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Add 3D visualization text
        cv2.putText(vis_frame, "3D Visualization", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # If we have tracking data, visualize tracks with 3D boxes
        if tracks is not None and tracks.shape[0] > 0:
            for track in tracks:
                xmin, ymin, zmin, xmax, ymax, zmax, track_id = track
                bbox = [xmin, ymin, xmax, ymax]
                # Calculate center for visualization
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                # Use average depth as z-coordinate
                center_z = (zmin + zmax) / 2
                
                # Use depth-based color (red=close, blue=far)
                max_depth = 10.0  # arbitrary scale for visualization
                normalized_depth = min(center_z / max_depth, 1.0)
                color = (0, int(255 * (1 - normalized_depth)), int(255 * normalized_depth))
                
                # Draw 3D bounding box
                vis_frame = self._draw_3d_box(vis_frame, bbox, center_z, color=color)
                
                # Draw track ID and depth
                cv2.putText(vis_frame, f"ID:{int(track_id)} Z:{center_z:.2f}m", 
                           (int(xmin), int(ymin) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw track center point
                cv2.circle(vis_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        else:
            # Otherwise visualize detections with 3D boxes
            for desc in descriptors:
                # Extract information
                x1, y1, x2, y2 = [int(coord) for coord in desc['bbox']]
                center_x, center_y, center_z = desc['center_3d']
                
                # Use depth-based color (red=close, blue=far)
                max_depth = 10.0  # arbitrary scale for visualization
                normalized_depth = min(center_z / max_depth, 1.0)
                color = (0, int(255 * (1 - normalized_depth)), int(255 * normalized_depth))
                
                # Draw 3D bounding box
                vis_frame = self._draw_3d_box(vis_frame, desc['bbox'], center_z, color=color)
                
                # Draw center point
                cv2.circle(vis_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                
                # Create label with depth
                coords_label = f"Z:{center_z:.2f}m"
                
                # Draw coordinates label
                cv2.putText(vis_frame, coords_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def _draw_3d_box(self, img, bbox, depth, color=(0, 255, 0), thickness=2):
        """
        Draw a 3D bounding box projection on a 2D image (internal helper method)
        
        Args:
            img: Image to draw on
            bbox: 2D box [x1, y1, x2, y2]
            depth: Depth value of the object (center_z)
            color: Box color
            thickness: Line thickness
        
        Returns:
            Image with 3D box drawn
        """
        x1, y1, x2, y2 = [int(i) for i in bbox]
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Estimate 3D box dimensions (in relative depth units)
        # Assuming average human dimensions (roughly height:width:depth = 7:2:1)
        real_height = 1.7  # relative units
        real_width = 0.5
        real_depth = 0.3
        
        # Rough estimate of focal length based on image size
        h, w = img.shape[:2]
        focal_length = w  # simplified approximation
        
        # Get center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate 3D corner points relative to center
        x3d = real_width / 2
        y3d = real_height / 2
        z3d = real_depth / 2
        
        # 3D corners in object space (front face then back face)
        corners_3d = np.array([
            [-x3d, -y3d, -z3d],  # front bottom left
            [+x3d, -y3d, -z3d],  # front bottom right
            [+x3d, +y3d, -z3d],  # front top right
            [-x3d, +y3d, -z3d],  # front top left
            [-x3d, -y3d, +z3d],  # back bottom left
            [+x3d, -y3d, +z3d],  # back bottom right
            [+x3d, +y3d, +z3d],  # back top right
            [-x3d, +y3d, +z3d],  # back top left
        ])
        
        # Scale corners based on the 2D box size and depth
        scale_x = width / real_width
        scale_y = height / real_height
        
        # Project 3D corners to 2D
        corners_2d = []
        for corner in corners_3d:
            # Simple perspective projection (z affects x,y)
            z_relative = corner[2] + depth
            x_projected = center_x + corner[0] * scale_x * depth / z_relative
            y_projected = center_y - corner[1] * scale_y * depth / z_relative  # Flip Y axis
            corners_2d.append((int(x_projected), int(y_projected)))
        
        # Convert to np array
        corners_2d = np.array(corners_2d, dtype=np.int32)
        
        # Draw bottom rectangle
        cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[1]), color, thickness)
        cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[2]), color, thickness)
        cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[3]), color, thickness)
        cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[0]), color, thickness)
        
        # Draw top rectangle
        cv2.line(img, tuple(corners_2d[4]), tuple(corners_2d[5]), color, thickness)
        cv2.line(img, tuple(corners_2d[5]), tuple(corners_2d[6]), color, thickness)
        cv2.line(img, tuple(corners_2d[6]), tuple(corners_2d[7]), color, thickness)
        cv2.line(img, tuple(corners_2d[7]), tuple(corners_2d[4]), color, thickness)
        
        # Draw vertical lines
        cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[4]), color, thickness)
        cv2.line(img, tuple(corners_2d[1]), tuple(corners_2d[5]), color, thickness)
        cv2.line(img, tuple(corners_2d[2]), tuple(corners_2d[6]), color, thickness)
        cv2.line(img, tuple(corners_2d[3]), tuple(corners_2d[7]), color, thickness)
        
        return img
        
    def visualize_depth_map(self, frame, colormap=cv2.COLORMAP_PLASMA):
        """
        Visualize the depth map
        
        Args:
            frame (numpy.ndarray): Input frame (unused, just for size reference)
            colormap: OpenCV colormap to use
            
        Returns:
            numpy.ndarray: Colorized depth map
        """
        if self.last_depth_map is None:
            return np.zeros_like(frame)
            
        # Normalize depth map to 0-255
        depth_map = self.last_depth_map.copy()
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap
        depth_vis = cv2.applyColorMap(depth_norm, colormap)
        
        return depth_vis