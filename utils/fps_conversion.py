import cv2

def convert_to_5fps(input_video_path, output_video_path):
    """
    Converts a video to 5 frames per second (fps) and saves the result to a new video file.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str): The path to the output video file.

    Raises:
        ValueError: If the input video file cannot be opened.
    """

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")

    # Get the frames per second (fps) of the input video
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the frame interval to achieve 5fps
    frame_interval = input_fps // 5

    # Get the video's frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, 5, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            out.write(frame)

        frame_count += 1

    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Video conversion completed.")
