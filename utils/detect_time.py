import cv2,pandas
class timer():
    def __init__(self,frame_list,roi,fps=5):
        self.frame_list=frame_list
        self.roi=roi
        self.fps=fps
    def check_roi(self,center_x,center_y):
        XMIN,YMIN,XMAX,YMAX=self.roi.astype(int)
        return XMIN<=center_x<=XMAX and YMIN<=center_y<=YMAX
    def put_time(self):
        t,inc=0,round(1/self.fps,2)
        storage=dict()
        result=[]
        for pair in self.frame_list:
            frame,df=pair
            for _,row in df.iterrows():
                xmin,ymin,xmax,ymax,person_id=row['xmin'],row['ymin'],row['xmax'],row['ymax'],row['id']
                center_x = (xmin + xmax)/2
                center_y= (ymin + ymax)/2
                if storage.get(person_id,'invalid') is 'invalid':
                        storage[person_id]=(t,self.check_roi(center_x,center_y))
                elif (self.check_roi(center_x,center_y)):
                        time_in=(t-storage[person_id][0])
                        cv2.putText(frame,text="Time in:"+str(time_in),org=(int(xmin),int(ymin)-15),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                            fontScale=1, color=(255, 0, 0),thickness=2)
                else:
                        storage[person_id]= (t,False)  
            result.append(frame)             
            t+=inc
        return result




        

