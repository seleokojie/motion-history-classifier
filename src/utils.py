import cv2


def annotate_frame(frame, label, position=(30,30)):
    """
    Annotates a video frame with a given label at a specified position.

    Args:
        frame (np.ndarray): The input video frame as a 2D or 3D NumPy array.
        label (str): The text label to annotate on the frame.
        position (tuple, optional): The (x, y) position of the label on the frame. Default is (30, 30).

    Returns:
        np.ndarray: The annotated video frame.
    """
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 1, cv2.LINE_AA)
    return frame