import torch
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add MiDaS to the path
MIDAS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../MiDaS'))
sys.path.append(MIDAS_PATH)

# Import MiDaS modules
from midas.model_loader import default_models, load_model

class MiDaSDepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize MiDaS depth estimator
        
        Args:
            model_type (str): MiDaS model type ('DPT_Large', 'DPT_Hybrid', or 'MiDaS_small')
        """
        # Load MiDaS model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model_type = model_type
        
        # Initialize MiDaS using torch hub
        print(f"Loading MiDaS model: {model_type}")
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            self.midas.to(self.device)
            self.midas.eval()
            
            # Initialize transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            print("MiDaS model loaded successfully")
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            print("Trying alternative loading method...")
            # Try alternative loading method
            if model_type == "MiDaS_small":
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            elif model_type == "DPT_Large":
                self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            elif model_type == "DPT_Hybrid":
                self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            else:
                # Default to small model
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                
            self.midas.to(self.device)
            self.midas.eval()
            
            # Initialize transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            print("MiDaS model loaded successfully with alternative method")
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image in BGR format (OpenCV default)
            
        Returns:
            numpy.ndarray: Depth map
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply input transforms
        input_batch = self.transform(img).to(self.device)
        
        # Prediction
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy array
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return depth_map 