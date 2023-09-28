# Display image and videos
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
from utils.Detector import Detector
from utils.properties import Properties,save_data
from utils.detect_time import timer
from utils.make_video import VideoEncoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load an official or custom model
model_1 = YOLO('models\yolov8n (1).pt')  # Load an official Detect model
model_2 = YOLO('models\yolov8n-pose.pt')  # Load an official Segment model
model_3 = YOLO('models\yolov8n-seg.pt')  # Load an official Pose model

# print("Model_1 info:",model_1.info(detailed=True),end='\n')
path="Shortit.mp4"
prop=Properties(path)
# prop.display_properties()
roi=np.array([400,200,850,600])
org_height,org_width,org_fps,org_num_frames=prop.height(),prop.width(),prop.fps(),prop.num_frames()
# print("Original height:{}, width:{}".format(org_height, org_width))
image_dir_path="outputs\image_directory"
df_dir_path="outputs\dataframe_dir"
detector=Detector(path,model=model_1,height=org_height,width=org_width,ROI=roi)
out_path="outputs"
frame_list=detector.detect()
save_data(data_list=frame_list,image_dir_path=image_dir_path,df_dir_path=df_dir_path)
Time=timer(frame_list,roi)
result=Time.put_time()
Video_maker=VideoEncoder(fps=5,width=org_width,height=org_height)
Video_maker.encode_frames(result,"outputs")










