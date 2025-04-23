import numpy as np
import cv2

# detect at import time
_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0

def compute_binary_sequence(frames, threshold=30, kernel_size=3):
    """
    Compute binary frame-difference masks Bt(x,y)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binaries = []
    if _GPU:
        # upload first frame
        prev = cv2.cuda_GpuMat(); prev.upload(frames[0])
        morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8U, kernel)

    for gray in frames[1:]:
            curr = cv2.cuda_GpuMat(); curr.upload(gray)
            diff = cv2.cuda.absdiff(curr, prev)
            _, bw = cv2.cuda.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
            opened = morph.apply(bw)
            binaries.append(opened.download())
            prev = curr
    else:
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i - 1])
            _, bw = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            binaries.append(bw)

    return binaries


def compute_mhi(binaries, tau=30):
    """
    Compute Motion History Image over binary masks
    """
    h, w = binaries[0].shape
    M = np.zeros((h, w), dtype=np.float32)
    for B in binaries:
        M[B == 1] = tau
        M[B == 0] = np.maximum(M[B == 0] - 1, 0)
    # normalize to [0,255]
    M_norm = np.uint8((M / tau) * 255)
    return M_norm
