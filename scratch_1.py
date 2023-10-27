# Display image and videos
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
from utils.Detector import Detector
from utils.properties import Properties, save_data, delete_directory
from utils.detect_time import Timer
from utils.make_video import VideoEncoder
from utils.Dataset import FrameExtractor
from utils.fps_conversion import convert_to_5fps
from utils.ROI_drawing import select_roi_from_video
import os

# Check if CUDA is available for GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO models
model_1 = YOLO('visitor-dwell-time\models\yolov8n (1).pt')  # Load an official Detect model
model_2 = YOLO('visitor-dwell-time\models\yolov8n-pose.pt')  # Load an official Segment model
model_3 = YOLO('visitor-dwell-time\models\yolov8n-seg.pt')  # Load an official Pose model

# Input video path
path = "oct-4-9-1230.mp4"

    

    

