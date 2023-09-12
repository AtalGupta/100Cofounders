import cv2

def show_properties(video_path):
    capture = cv2.VideoCapture(video_path)
    print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
    print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
    print("CAP_PROP_FRAME_COUNT  : '{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("CAP_PROP_BRIGHTNESS : '{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))
    print("CAP_PROP_CONTRAST : '{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)))
    print("CAP_PROP_SATURATION : '{}'".format(capture.get(cv2.CAP_PROP_SATURATION)))
    print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
    print("CAP_PROP_GAIN  : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
    print("CAP_PROP_CONVERT_RGB : '{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))

def make_parallelogram(frame,p1,p2,p3,p4,color=(255,0,0)):
    cv2.line(frame,p1,p2,color,5)
    cv2.line(frame,p2,p3,color,5)
    cv2.line(frame,p3,p4,color,5)
    cv2.line(frame,p4,p1,color,5)
    return frame

def check_roi(center_x,center_y,ROI):
    XMIN,YMIN,XMAX,YMAX=ROI.astype(int)
    
    return XMIN<=center_x<=XMAX and YMIN<=center_y<=YMAX


