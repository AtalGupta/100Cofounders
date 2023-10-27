import os
from utils.properties import delete_directory, create_directory,check_roi
from PIL import Image
import pandas as pd

class FrameExtractor:
    """
    A class for extracting frames from images based on a provided DataFrame.

    Args:
        org_dir (str): The directory containing the original images.
        dataframe_dir (str): The directory containing DataFrames with bounding box coordinates.
        output_dir (str): The directory where extracted frames will be saved.

    Methods:
        extract_frames:
            Extracts frames from images based on bounding box coordinates in DataFrames.
    """

    def __init__(self, org_dir, dataframe_dir, output_dir,roi):
        """
        Initializes a FrameExtractor instance.

        Args:
            org_dir (str): The directory containing the original images.
            dataframe_dir (str): The directory containing DataFrames with bounding box coordinates.
            output_dir (str): The directory where extracted frames will be saved.
        """
        self.org_dir = org_dir
        self.dataframe_dir = dataframe_dir
        self.output_dir = output_dir
        self.roi=roi

    def extract_frames(self,vid_name=""):
        """
        Extracts frames from images based on bounding box coordinates in DataFrames.

        This method loops through each DataFrame and extracts frames based on the
        bounding box coordinates, saving them to the specified output directory.
        """
        df_dir = os.listdir(self.dataframe_dir)
        img_dir = os.listdir(self.org_dir)

        # Delete the existing output directory if it exists, then create a new one.
        delete_directory(self.output_dir)
        create_directory(self.output_dir)

        num = 0
        for ele in range(len(df_dir)):
            img = Image.open(os.path.join(self.org_dir, img_dir[ele]))
            df = pd.read_csv(os.path.join(self.dataframe_dir, df_dir[ele]))

            for _, row in df.iterrows():
                xmin, ymin, xmax, ymax, person_id = (
                    row['xmin'],
                    row['ymin'],
                    row['xmax'],
                    row['ymax'],
                    int(row['id'])
                )
                box = (xmin, ymin, xmax, ymax)
                center_x,center_y=(xmin+xmax)/2,(ymin+ymax)/2
                if(check_roi(center_x,center_y,ROI=self.roi)):
                    img2 = img.crop(box)
                    file_name = vid_name+"_"+str(num) + "_" + str(person_id) + ".jpg"
                    file_path = os.path.join(self.output_dir, file_name)
                    img2.save(file_path)
                    num += 1

# Usage example:
# extractor = FrameExtractor("outputs/org_dir", "outputs/dataframe_dir", "outputs/dataset")
# extractor.extract_frames()
