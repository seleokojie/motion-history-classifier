import cv2


def annotate_frame(frame, label, position=(30,30)):
    """
    Draw label on frame
    """
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0,255,0), 2, cv2.LINE_AA)
    return frame