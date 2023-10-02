import cv2

def select_roi_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        raise ValueError("Error: Could not open video file.")

    # Create a window and display the video
    cv2.namedWindow("Select ROI")
    cv2.imshow("Select ROI", frame)

    # Initialize variables to store ROI coordinates
    x1, y1, x2, y2 = -1, -1, -1, -1
    roi_selected = False

    def select_roi(event, x, y, flags, param):
        nonlocal x1, y1, x2, y2, roi_selected

        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x, y
            roi_selected = False

        elif event == cv2.EVENT_LBUTTONUP:
            x2, y2 = x, y
            roi_selected = True
            # Draw a rectangle around the selected ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Select ROI", frame)

    cv2.setMouseCallback("Select ROI", select_roi)

    while not roi_selected:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Select ROI", frame)

        if cv2.waitKey(100) & 0xFF == 27:
            break

    # Close the video window
    cv2.destroyWindow("Select ROI")

    # Release the video capture object
    cap.release()

    # Return ROI coordinates
    return [x1, y1, x2, y2]


