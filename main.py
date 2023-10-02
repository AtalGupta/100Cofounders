# Display image and videos
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
from utils.Detector import Detector
from utils.properties import Properties,save_data,delete_directory
from utils.detect_time import timer
from utils.make_video import VideoEncoder
from utils.Dataset import FrameExtractor
from utils.fps_conversion import convert_to_5fps
from utils.ROI_drawing import select_roi_from_video
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load an official or custom model
model_1 = YOLO('models\yolov8n (1).pt')  # Load an official Detect model
model_2 = YOLO('models\yolov8n-pose.pt')  # Load an official Segment model
model_3 = YOLO('models\yolov8n-seg.pt')  # Load an official Pose model

# print("Model_1 info:",model_1.info(detailed=True),end='\n')
input_path="Shortit.mp4"
convert_to_5fps(input_path, "5fps_video.mp4")
path="5fps_video.mp4"
prop=Properties(path)
# prop.display_properties()
roi=np.array(select_roi_from_video(path))
org_height,org_width,org_fps,org_num_frames=prop.height(),prop.width(),prop.fps(),prop.num_frames()
# print("Original height:{}, width:{}".format(org_height, org_width))
image_dir_path="outputs\image_directory"
df_dir_path="outputs\dataframe_dir"
delete_directory(image_dir_path)
delete_directory(df_dir_path)
num_frame=100
out_path=os.path.join("outputs","org_dir")
detector=Detector(path,model=model_1,height=org_height,width=org_width,ROI=roi,num_frames=num_frame,
                  save_frames=True,out_path=out_path)
out_path="outputs"
frame_list=detector.detect()
save_data(data_list=frame_list,image_dir_path=image_dir_path,df_dir_path=df_dir_path)
Time=timer(frame_list,roi)
result=Time.put_time()
Video_maker=VideoEncoder(fps=5,width=org_width,height=org_height)
Video_maker.encode_frames(result,"outputs")

fext=FrameExtractor(org_dir="outputs\org_dir",dataframe_dir="outputs\dataframe_dir",
                    output_dir="outputs\dataset")
fext.extract_frames()








