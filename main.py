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
path = "video_1\out02.mp4"

# Convert input video to 5fps and save it
# convert_to_5fps(input_path, "5fps_video.mp4")
# path = "5fps_video.mp4"

# Initialize properties of the video
prop = Properties(path)
# Uncomment the following line to display video properties
# prop.display_properties()
roi = np.array(select_roi_from_video(path))  # Select a Region of Interest (ROI)
org_height, org_width, org_fps, org_num_frames = prop.height(), prop.width(), prop.fps(), prop.num_frames()
# Uncomment the following line to print original height and width
# print("Original height: {}, width: {}".format(org_height, org_width))

# Define paths for saving image frames and DataFrames
image_dir_path = "visitor-dwell-time\\temp_output\image_directory"
df_dir_path = "visitor-dwell-time\\temp_output\dataframe_directory"

# Delete existing directories if they exist
delete_directory(image_dir_path)
delete_directory(df_dir_path)

num_frame = org_num_frames # Number of frames to process
out_path = "visitor-dwell-time\\temp_output\orginal_directory"

# Initialize the detector to detect and track objects in the video
detector = Detector(path, model=model_1, height=org_height, width=org_width, ROI=roi,
                    num_frames=num_frame, save_frames=True, out_path=out_path)

# Detect and track objects in the video
frame_list = detector.detect()

# Save detected frames and DataFrames
save_data(data_list=frame_list, image_dir_path=image_dir_path, df_dir_path=df_dir_path)

# # Create a timer to measure the time spent by objects in the ROI
# Time = Timer(frame_list, roi)

# # Calculate and display the time spent by objects in the ROI
# result = Time.put_time()

# # Encode the annotated frames into a new video
# Video_maker = VideoEncoder(fps=5, width=org_width, height=org_height)
# Video_maker.encode_frames(result, "outputs")

# Extract frames from the original video based on the provided DataFrames
fext = FrameExtractor(org_dir=out_path, dataframe_dir=df_dir_path,
                    output_dir="generated_data\data_2",roi=roi)

# Execute the frame extraction process
fext.extract_frames(vid_name="2")
delete_directory(out_path)
