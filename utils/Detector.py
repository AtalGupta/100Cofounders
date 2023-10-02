import os,tqdm,cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
class Detector():
    def __init__(self,video_path,ROI,model,num_frames=100,fps=5,height=650
                 ,width=800,save_frames=False,
                 out_path=""
                 ):
        self.path=video_path
        self.roi=ROI #should be of type numpy
        self.model=model
        self.height=height
        self.width=width
        self.fps=fps
        self.num_frames=num_frames
        self.save=save_frames
        self.out_path=out_path
        
    def detect(self):
        video=cv2.VideoCapture(self.path)
        frame_list=[] #type -> frame : (track values)
        cnt=0
        for i in tqdm(range(self.num_frames), ncols=200):
            _,frame=video.read()
            result=self.model.track(frame,persist=True)

            if (self.save):
                os.makedirs(self.out_path, exist_ok=True)
                frame_filename = os.path.join(self.out_path, f"frame_{cnt:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                cnt+=1
            annotated_frame=result[0].plot()
            XMIN,YMIN,XMAX,YMAX=self.roi.astype(int)
            cv2.rectangle(annotated_frame,(XMIN,YMIN),(XMAX,YMAX),(0,255,0),5)
            df=pd.DataFrame(result[0].cpu().numpy().boxes.data
                    ,columns=['xmin', 'ymin', 'xmax', 'ymax','id','conf','class'])
            
            curr=(annotated_frame,df)
            frame_list.append(curr)
        return frame_list
            
