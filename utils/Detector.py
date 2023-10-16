import os
import tqdm
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

class Detector:
    """
    A class for detecting and tracking objects in a video.

    Args:
        video_path (str): The path to the video file.
        ROI (numpy.ndarray): The Region of Interest (ROI) as a NumPy array [XMIN, YMIN, XMAX, YMAX].
        model (ultralytics.YOLO): The YOLO model for object detection and tracking.
        num_frames (int, optional): Number of frames to process, default is 100.
        fps (int, optional): Frames per second, default is 5.
        height (int, optional): Height of the frames, default is 650.
        width (int, optional): Width of the frames, default is 800.
        save_frames (bool, optional): Whether to save annotated frames, default is False.
        out_path (str, optional): Output directory for saving annotated frames, required if save_frames is True.

    Methods:
        detect:
            Processes the video frames, detects and tracks objects, and returns annotated frames with DataFrames.
    """

    def __init__(self, video_path, ROI, model, num_frames=100, fps=5, height=650, width=800, save_frames=False, out_path=""):
        """
        Initializes a Detector instance.

        Args:
            video_path (str): The path to the video file.
            ROI (numpy.ndarray): The Region of Interest (ROI) as a NumPy array [XMIN, YMIN, XMAX, YMAX].
            model (ultralytics.YOLO): The YOLO model for object detection and tracking.
            num_frames (int, optional): Number of frames to process, default is 100.
            fps (int, optional): Frames per second, default is 5.
            height (int, optional): Height of the frames, default is 650.
            width (int, optional): Width of the frames, default is 800.
            save_frames (bool, optional): Whether to save annotated frames, default is False.
            out_path (str, optional): Output directory for saving annotated frames, required if save_frames is True.
        """
        self.video_path = video_path
        self.ROI = ROI
        self.model = model
        self.height = height
        self.width = width
        self.fps = fps
        self.num_frames = num_frames
        self.save = save_frames
        self.out_path = out_path

    def detect(self):
        """
        Processes video frames, detects and tracks objects, and returns annotated frames with DataFrames.

        This method iterates through the video frames, performs object detection and tracking,
        and annotates the frames with bounding boxes. It also saves annotated frames if specified.

        Returns:
            list: A list of frame and DataFrame pairs.
        """
        video = cv2.VideoCapture(self.video_path)
        frame_list = []  # type -> frame : (track values)
        cnt = 0
        for i in tqdm(range(self.num_frames), ncols=200):
            _, frame = video.read()
            result = self.model.track(frame, persist=True)

            if self.save:
                os.makedirs(self.out_path, exist_ok=True)
                frame_filename = os.path.join(self.out_path, f"frame_{cnt:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                cnt += 1

            annotated_frame = result[0].plot()
            XMIN, YMIN, XMAX, YMAX = self.ROI.astype(int)
            cv2.rectangle(annotated_frame, (XMIN, YMIN), (XMAX, YMAX), (0, 255, 0), 5)
            df = pd.DataFrame(result[0].cpu().numpy().boxes.data,
                              columns=['xmin', 'ymin', 'xmax', 'ymax', 'id', 'conf', 'class'])

            curr = (annotated_frame, df)
            frame_list.append(curr)

        return frame_list
