import cv2
import os
import shutil
class Properties:
    def __init__(self,video_path):
        self.capture=cv2.VideoCapture(video_path)
    def display_properties(self):
        print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("CAP_PROP_FPS : '{}'".format(self.capture.get(cv2.CAP_PROP_FPS)))
        print("CAP_PROP_POS_MSEC : '{}'".format(self.capture.get(cv2.CAP_PROP_POS_MSEC)))
        print("CAP_PROP_FRAME_COUNT  : '{}'".format(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("CAP_PROP_BRIGHTNESS : '{}'".format(self.capture.get(cv2.CAP_PROP_BRIGHTNESS)))
        print("CAP_PROP_CONTRAST : '{}'".format(self.capture.get(cv2.CAP_PROP_CONTRAST)))
        print("CAP_PROP_SATURATION : '{}'".format(self.capture.get(cv2.CAP_PROP_SATURATION)))
        print("CAP_PROP_HUE : '{}'".format(self.capture.get(cv2.CAP_PROP_HUE)))
        print("CAP_PROP_GAIN  : '{}'".format(self.capture.get(cv2.CAP_PROP_GAIN)))
        print("CAP_PROP_CONVERT_RGB : '{}'".format(self.capture.get(cv2.CAP_PROP_CONVERT_RGB)))
    def height(self):
        return float(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    def width(self):
        return float(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    def fps(self):
        return int(self.capture.get(cv2.CAP_PROP_FPS))
    def num_frames(self):
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

def check_roi(center_x,center_y,ROI):
    XMIN,YMIN,XMAX,YMAX=ROI.astype(int)
    
    return XMIN<=center_x<=XMAX and YMIN<=center_y<=YMAX

def create_directory(directory_path):
    try:
        os.mkdir(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")
    except Exception as e:
        print(f"An error occurred while creating directory '{directory_path}': {str(e)}")

def save_data(data_list,image_dir_path,df_dir_path):
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

    :param directory_path: The path of the directory to be deleted.
    """
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' does not exist.")







