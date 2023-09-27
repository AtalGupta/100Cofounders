import os
import cv2

class VideoEncoder:
    def __init__(self, fps=5, width=800, height=650):
        self.fps = fps
        self.width = int(width)
        self.height = int(height)

    def encode_frames(self, frame_list, out_path):
        video_name = 'result.mp4'
        output_path = os.path.join(out_path, video_name)
        VIDEO_CODEC = "MP4V"
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)  # Create the codec
        output_video = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        for frame in frame_list:
            output_video.write(frame)
        output_video.release()

