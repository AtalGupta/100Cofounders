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

main_video="video_1\out00.mp4"
roi = np.array(select_roi_from_video(main_video))  # Select a Region of Interest (ROI)

for relative in os.listdir("video_1"):
    path=os.path.join("video_1",relative)
    # Initialize properties of the video
    prop = Properties(path)
    org_height, org_width, org_fps, org_num_frames = prop.height(), prop.width(), prop.fps(), prop.num_frames()

    # Define paths for saving image frames and DataFrames
    image_dir_path = "visitor-dwell-time\\temp_output\image_dir"
    df_dir_path = "visitor-dwell-time\\temp_output\df_dir"
    org_dir_path="visitor-dwell-time\\temp_output\org_dir"

    # Delete existing directories if they exist
    delete_directory(image_dir_path)
    delete_directory(df_dir_path)
    delete_directory(org_dir_path)

    detector = Detector(path, model=model_1, height=org_height, width=org_width, ROI=roi,
                    num_frames=org_num_frames,save_frames=True,out_path=org_dir_path)

    frame_list = detector.detect()
    # Save detected frames and DataFrames
    save_data(data_list=frame_list, image_dir_path=image_dir_path, df_dir_path=df_dir_path)
    # Extract frames from the original video based on the provided DataFrames
    fext = FrameExtractor(org_dir=org_dir_path, dataframe_dir=df_dir_path,
                        output_dir="generated_data\data_1",roi=roi)

    # Execute the frame extraction process
    fext.extract_frames(vid_name=relative)
    print(f"Successfully saved the for video {relative}")
    

    

