import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from utils.Detector import Detector
from utils.properties import Properties, save_data, delete_directory,check_roi
from utils.detect_time import Timer
from utils.make_video import VideoEncoder
from utils.Dataset import FrameExtractor
from utils.fps_conversion import convert_to_5fps
from utils.ROI_drawing import select_roi_from_video
import os
import pandas as pd
from tqdm import tqdm
# Check if CUDA is available for GPU acceleration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO models
model_1 = YOLO('visitor-dwell-time\models\yolov8n (1).pt')  # Load an official Detect model
model_2 = YOLO('visitor-dwell-time\models\yolov8n-pose.pt')  # Load an official Segment model
model_3 = YOLO('visitor-dwell-time\models\yolov8n-seg.pt')  # Load an official Pose model

# Input video path
path = "oct-4-9-1230.mp4"
vid_name="xyz"
prop = Properties(path)
roi = np.array(select_roi_from_video(path))
org_height, org_width, org_fps, org_num_frames = prop.height(), prop.width(), prop.fps(), prop.num_frames()
output_dir="generated_data\data_xd_1"
video = cv2.VideoCapture(path)
num_frames=org_num_frames
num=0
for i in tqdm(range(num_frames),ncols=200):
    _, frame = video.read()
    result = model_1.track(frame,classes=[0],persist=True)
    try:
        df = pd.DataFrame(result[0].cpu().numpy().boxes.data,
                                columns=['xmin', 'ymin', 'xmax', 'ymax', 'id', 'conf'])
    except:
        print("Skipping frame")
        continue
    for _, row in df.iterrows():
            xmin, ymin, xmax, ymax, person_id = (
                    int(row['xmin']),
                    int(row['ymin']),
                    int(row['xmax']),
                    int(row['ymax']),
                    int(row['id'])
                )
            box = (xmin, ymin, xmax, ymax)
            center_x,center_y=(xmin+xmax)/2,(ymin+ymax)/2
            if(check_roi(center_x,center_y,ROI=roi)):
                    img2=frame[xmin:xmax,ymin:ymax]
                    file_name = vid_name+"_"+str(num) + "_" + str(person_id) + ".jpg"
                    file_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(file_path,img2)
                    print("file saved")
                    num += 1
            else:
                  print("Detected person not in roi")
