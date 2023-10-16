import cv2
import os
import shutil

class Properties:
    """
    A class for displaying properties of a video file.

    Args:
        video_path (str): The path to the video file.

    Methods:
        display_properties:
            Displays various properties of the video.
        height:
            Returns the height of the video frames.
        width:
            Returns the width of the video frames.
        fps:
            Returns the frames per second (fps) of the video.
        num_frames:
            Returns the total number of frames in the video.
    """

    def __init__(self, video_path):
        """
        Initializes a Properties instance for a given video file.

        Args:
            video_path (str): The path to the video file.
        """
        self.capture = cv2.VideoCapture(video_path)

    def display_properties(self):
        """
        Displays various properties of the video.
        """
        # Display video properties
        print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("CV_CAP_PROP_FRAME_HEIGHT: '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("CAP_PROP_FPS: '{}'".format(self.capture.get(cv2.CAP_PROP_FPS)))
        print("CAP_PROP_POS_MSEC: '{}'".format(self.capture.get(cv2.CAP_PROP_POS_MSEC)))
        print("CAP_PROP_FRAME_COUNT: '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("CAP_PROP_BRIGHTNESS: '{}'".format(self.capture.get(cv2.CAP_PROP_BRIGHTNESS)))
        print("CAP_PROP_CONTRAST: '{}'".format(self.capture.get(cv2.CAP_PROP_CONTRAST)))
        print("CAP_PROP_SATURATION: '{}'".format(self.capture.get(cv2.CAP_PROP_SATURATION)))
        print("CAP_PROP_HUE: '{}'".format(self.capture.get(cv2.CAP_PROP_HUE)))
        print("CAP_PROP_GAIN: '{}'".format(self.capture.get(cv2.CAP_PROP_GAIN)))
        print("CAP_PROP_CONVERT_RGB: '{}'".format(self.capture.get(cv2.CAP_PROP_CONVERT_RGB)))

    def height(self):
        """
        Returns the height of the video frames.

        Returns:
            float: The height of the video frames.
        """
        return float(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def width(self):
        """
        Returns the width of the video frames.

        Returns:
            float: The width of the video frames.
        """
        return float(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    def fps(self):
        """
        Returns the frames per second (fps) of the video.

        Returns:
            int: The frames per second of the video.
        """
        return int(self.capture.get(cv2.CAP_PROP_FPS))

    def num_frames(self):
        """
        Returns the total number of frames in the video.

        Returns:
            int: The total number of frames in the video.
        """
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

def check_roi(center_x, center_y, ROI):
    """
    Checks if a point is within a defined Region of Interest (ROI).

    Args:
        center_x (int): X-coordinate of the point.
        center_y (int): Y-coordinate of the point.
        ROI (numpy.ndarray): The Region of Interest (ROI) as a NumPy array [XMIN, YMIN, XMAX, YMAX].

    Returns:
        bool: True if the point is within the ROI, False otherwise.
    """
    XMIN, YMIN, XMAX, YMAX = ROI.astype(int)
    return XMIN <= center_x <= XMAX and YMIN <= center_y <= YMAX

def create_directory(directory_path):
    """
    Creates a directory if it doesn't exist.

    Args:
        directory_path (str): The path of the directory to be created.
    """
    try:
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")
    except Exception as e:
        print(f"An error occurred while creating directory '{directory_path}': {str(e)}")

def save_data(data_list, image_dir_path, df_dir_path):
    """
    Saves image frames and DataFrames to specified directories.

    Args:
        data_list (list): A list of (image, DataFrame) pairs to be saved.
        image_dir_path (str): The directory path to save image frames.
        df_dir_path (str): The directory path to save DataFrames.
    """
    try:
        # Create the directories if they don't exist
        os.makedirs(image_dir_path, exist_ok=True)
        os.makedirs(df_dir_path, exist_ok=True)

        for i, (image, df) in enumerate(data_list):
            image_filename = os.path.join(image_dir_path, f"image_{i}.jpg")
            cv2.imwrite(image_filename, image)
            dataframe_filename = os.path.join(df_dir_path, f"dataframe_{i}.csv")
            df.to_csv(dataframe_filename, index=False)

            print(f"Saved image {i} to '{image_filename}'")
            print(f"Saved DataFrame {i} to '{dataframe_filename}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def delete_directory(directory_path):
    """
    Deletes a directory and its contents if it exists.

    Args:
        directory_path (str): The path of the directory to be deleted.
    """
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' does not exist.")
