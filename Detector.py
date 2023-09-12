import os,tqdm,cv2
import pandas as pd
import time
from utils import check_roi
class Detector():
    def __init__(self,video_path,ROI,model,num_frames,fps,height,width):
        self.path=video_path
        self.roi=ROI #should be of type numpy
        self.model=model
        self.height=height
        self.width=width
        self.fps=fps
        self.num_frames=num_frames
    def video_encoder(self,frame_list,out_path):
        video_name = 'result.mp4'
        output_path = os.path.join(out_path,video_name) 
        VIDEO_CODEC = "MP4V"
        output_video=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*VIDEO_CODEC),self.fps
                                     ,(self.width, self.height))
        for frame in frame_list:
            output_video.write(frame)
        output_video.release()
    def detect(self):
        video=cv2.VideoCapture(self.path)
        frame_list=[]
        storage=dict()
        for i in tqdm(range(self.num_frames)):
            _,frame=video.read()
            result=self.model.track(frame,persist=True)
            annotated_frame=result[0].plot()
            XMIN,YMIN,XMAX,YMAX=self.roi.astype(int)
            cv2.rectangle(annotated_frame,(XMIN,YMIN),(XMAX,YMAX),(0,255,0),5)
            track_ids = result[0].boxes.id.int().cpu().tolist()
            df=pd.DataFrame(result[0].cpu().numpy().boxes.data
                    ,columns=['xmin', 'ymin', 'xmax', 'ymax','id','conf','class'])
            
            for index, row in df.iterrows():
                xmin,ymin,xmax,ymax=row['xmin'],row['ymin'],row['xmax'],row['ymax']
                idx=row['id']
                center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
                
                if storage.get(idx,'invalid') is 'invalid':
                    storage[idx]=(time.time(),check_roi(center_x,center_y,ROI))
                elif check_roi(center_x,center_y,self.roi):
                    time_in=round(time.time()-storage[idx][0],2)
                    cv2.putText(annotated_frame,text="Time in:"+str(time_in),org=(int(xmin),int(ymin)-15),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                            fontScale=1, color=(255, 0, 0),thickness=2)
                else:
                    storage[idx]=(time.time(),False)
                    
            
            frame_list.append(annotated_frame)

