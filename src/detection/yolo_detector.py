import torch
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOv5Detector:
    def __init__(self, model_name="yolov8s", conf_threshold=0.35, iou_threshold=0.45, classes=[0]):
        """
        Initialize YOLOv5 detector
        
        Args:
            model_name (str): YOLOv5 model name ('yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt')
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            classes (list): List of classes to detect, e.g. [0] for persons only (None for all classes)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        
        # Load YOLOv5 model using ultralytics
        print(f"Loading YOLO model: {model_name}")
        
        # For YOLO v5 models
        if not model_name.startswith("yolov5"):
            self.model = YOLO(model_name)
        else:
            # For YOLOv5 models, use the pretrained model from ultralytics
            self.model = YOLO("yolov8s")  # Fallback to YOLOv8s
            print(f"YOLOv5 models are not directly supported in the latest ultralytics package.")
            print(f"Using YOLOv8s model instead.")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Print class filtering info
        if self.classes is not None:
            print(f"Filtering detections to classes: {self.classes}")
    
    def detect(self, image):
        """
        Detect objects in an image
        
        Args:
            image (numpy.ndarray): Input image in BGR format (OpenCV default)
            
        Returns:
            list: List of detections, each detection is a dictionary with keys:
                - 'bbox': [x1, y1, x2, y2] (coordinates of the bounding box)
                - 'confidence': Detection confidence
                - 'class_id': Class ID
                - 'class_name': Class name
        """
        # Run inference
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device, classes=self.classes)
        
        # Process results
        detections = []
        
        # Get predictions from the first image
        result = results[0]
        
        # Extract boxes, confidences, and class IDs
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence
            conf = float(box.conf[0].cpu().numpy())
            
            # Get class ID and name
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = result.names[cls_id]
            
            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': conf,
                'class_id': cls_id,
                'class_name': cls_name
            }
            
            detections.append(detection)
        
        return detections 