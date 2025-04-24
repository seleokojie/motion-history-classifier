try:
    import cupyx as cp
    from cupyx.scipy.ndimage import binary_opening
    # ensure nvrtc & CUDA are really available
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _HAS_GPU = False

import numpy as np
import cv2

def compute_binary_sequence(frames, threshold=30, kernel_size=3):
    """
    Compute binary frame‐difference masks Bt(x,y) on the GPU via CuPy,
    then download them back to NumPy for downstream processing.
    """
    if _HAS_GPU:
        kernel = cp.ones((kernel_size, kernel_size), cp.uint8)
        binaries = []
        for i in range(1, len(frames)):
            a = cp.asarray(frames[i-1], dtype=cp.uint8)
            b = cp.asarray(frames[i],   dtype=cp.uint8)
            diff = cp.abs(b - a)
            bw = (diff >= threshold).astype(cp.uint8)
            opened = binary_opening(bw, structure=kernel)
            binaries.append(cp.asnumpy(opened))
        return binaries

    # CPU fallback
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binaries = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
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
