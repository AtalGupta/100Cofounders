import os
from utils.properties import delete_directory, create_directory
from PIL import Image
import pandas as pd

class FrameExtractor:
    def __init__(self, org_dir, dataframe_dir, output_dir):
        self.org_dir = org_dir
        self.dataframe_dir = dataframe_dir
        self.output_dir = output_dir

    def extract_frames(self):
        df_dir = os.listdir(self.dataframe_dir)
        img_dir = os.listdir(self.org_dir)
        
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
                img2 = img.crop(box)
                file_name = str(num) + "_" + str(person_id) + ".jpg"
                file_path = os.path.join(self.output_dir, file_name)
                img2.save(file_path)
                num += 1

# Usage example:
# extractor = FrameExtractor("outputs/org_dir", "outputs/dataframe_dir", "outputs/dataset")
# extractor.extract_frames()
