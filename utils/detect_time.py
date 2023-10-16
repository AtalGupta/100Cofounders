import cv2
import pandas

class Timer:
    """
    A class for tracking and displaying time-in for individuals in frames.

    Args:
        frame_list (list): A list of frame and DataFrame pairs.
        roi (tuple): A tuple representing the Region of Interest (ROI) as (XMIN, YMIN, XMAX, YMAX).
        fps (int, optional): Frames per second, default is 5.

    Methods:
        check_roi(center_x, center_y):
            Checks if a given point is within the defined ROI.

        put_time():
            Calculates and displays time-in for individuals within the ROI in frames.
    """

    def __init__(self, frame_list, roi, fps=5):
        """
        Initializes a Timer instance.

        Args:
            frame_list (list): A list of frame and DataFrame pairs.
            roi (tuple): A tuple representing the Region of Interest (ROI) as (XMIN, YMIN, XMAX, YMAX).
            fps (int, optional): Frames per second, default is 5.
        """
        self.frame_list = frame_list
        self.roi = roi
        self.fps = fps

    def check_roi(self, center_x, center_y):
        """
        Checks if a given point is within the defined ROI.

        Args:
            center_x (int): X-coordinate of the point.
            center_y (int): Y-coordinate of the point.

        Returns:
            bool: True if the point is within the ROI, False otherwise.
        """
        XMIN, YMIN, XMAX, YMAX = self.roi
        return XMIN <= center_x <= XMAX and YMIN <= center_y <= YMAX

    def put_time(self):
        """
        Calculates and displays time-in for individuals within the ROI in frames.

        This method iterates through the provided frame and DataFrame pairs, calculates
        and displays the time-in for individuals within the ROI, and returns the resulting frames.
        """
        t, inc = 0, round(1 / self.fps, 2)
        storage = dict()
        result = []
        for pair in self.frame_list:
            frame, df = pair
            for _, row in df.iterrows():
                xmin, ymin, xmax, ymax, person_id = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['id']
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                if storage.get(person_id, 'invalid') is 'invalid':
                    storage[person_id] = (t, self.check_roi(center_x, center_y))
                elif self.check_roi(center_x, center_y):
                    time_in = (t - storage[person_id][0])
                    cv2.putText(frame, text="Time in:" + str(time_in), org=(int(xmin), int(ymin) - 15),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1, color=(255, 0, 0), thickness=2)
                else:
                    storage[person_id] = (t, False)
            result.append(frame)
            t += inc
        return result
