# Display image and videos
import cv2
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL.ImageDraw import Draw
import os
from ultralytics import YOLO
from Detector import Detector
from utils import show_properties

# Load an official or custom model
model_1 = YOLO('D:\Multi_camera_tracker\yolov8n (1).pt')  # Load an official Detect model
model_2 = YOLO('D:\Multi_camera_tracker\yolov8n-pose.pt')  # Load an official Segment model
model_3 = YOLO('D:\Multi_camera_tracker\yolov8n-pose.pt')  # Load an official Pose model

ROI_1=np.array([400,200,850,600]) # ROI of the given 
video_path_1="D:\Multi_camera_tracker\Shortit.mp4"

fps=5
num_frames=100
height,width,_,__=show_properties(video_path_1)
object_1=Detector(video_path_1,ROI_1,model_3,height,width)
output_path="D:\Multi_camera_tracker"
object_1.detect(output_path)



