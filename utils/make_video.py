import os
import cv2

class VideoEncoder:
    """
    A class for encoding a list of frames into a video.

    Args:
        fps (int, optional): Frames per second, default is 5.
        width (int, optional): Width of the video frames, default is 800.
        height (int, optional): Height of the video frames, default is 650.

    Methods:
        encode_frames(frame_list, out_path):
            Encodes a list of frames into a video and saves it at the specified output path.
    """

    def __init__(self, fps=5, width=800, height=650):
        """
        Initializes a VideoEncoder instance.

        Args:
            fps (int, optional): Frames per second, default is 5.
            width (int, optional): Width of the video frames, default is 800.
            height (int, optional): Height of the video frames, default is 650.
        """
        self.fps = fps
        self.width = int(width)
        self.height = int(height)

    def encode_frames(self, frame_list, out_path):
        """
        Encodes a list of frames into a video and saves it at the specified output path.

        Args:
            frame_list (list): A list of frames to be encoded into a video.
            out_path (str): The path where the output video will be saved.
        """
        video_name = 'result.mp4'
        output_path = os.path.join(out_path, video_name)
        VIDEO_CODEC = "MP4V"
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)  # Create the codec
        output_video = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        for frame in frame_list:
            output_video.write(frame)
        output_video.release()
