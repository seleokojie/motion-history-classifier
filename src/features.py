try:
    import cupyx as cp
    from cupyx.scipy.ndimage import binary_opening
    # ensure nvrtc & CUDA are really available
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _HAS_GPU = False

import numpy as np

def central_moments(img, p, q):
    """
    Computes the central moment \(\mu_{pq}\) of an image.

    Args:
        img (np.ndarray): Input image as a 2D NumPy array.
        p (int): Order of the moment along the x-axis.
        q (int): Order of the moment along the y-axis.

    Returns:
        float: The computed central moment value.
    """
    h, w = img.shape
    Y, X = np.mgrid[:h, :w]
    I = img.astype(np.float64)
    m00 = I.sum()
    if m00 == 0:
        return 0.0
    x_bar = (X * I).sum() / m00
    y_bar = (Y * I).sum() / m00
    return (((X - x_bar)**p * (Y - y_bar)**q * I).sum())


def scale_invariant(mu_pq, mu_00, p, q):
    """
    Computes the scale-invariant moment based on central moments.

    Args:
        mu_pq (float): Central moment \(\mu_{pq}\).
        mu_00 (float): Zeroth central moment \(\mu_{00}\).
        p (int): Order of the moment along the x-axis.
        q (int): Order of the moment along the y-axis.

    Returns:
        float: The scale-invariant moment value.
    """
    return mu_pq / (mu_00 ** ((p + q) / 2 + 1)) if mu_00 > 0 else 0.0


def extract_hu_features(img: np.ndarray) -> np.ndarray:
    """
    Computes the 8 scale-invariant central moments (Hu moments) of an image.

    This function uses GPU acceleration via CuPy if available, otherwise falls back to CPU computation.

    Args:
        img (np.ndarray): Input image as a 2D NumPy array.

    Returns:
        np.ndarray: A NumPy array of shape (8,) containing the computed Hu moments.
    """
    if _HAS_GPU:
        # GPU path with CuPy
        I = cp.asarray(img, dtype=cp.float64)
        h, w = I.shape
        Y, X = cp.mgrid[:h, :w]

        m00 = I.sum()
        if m00 == 0:
            return np.zeros(8, dtype=np.float32)

        x_bar = (X * I).sum() / m00
        y_bar = (Y * I).sum() / m00

        feats = []
        for p, q in [(2, 0), (1, 1), (0, 2), (3, 0),
                     (2, 1), (1, 2), (0, 3), (2, 2)]:
            mu = (I * (X - x_bar) ** p * (Y - y_bar) ** q).sum()
            nu = mu / (m00 ** ((p + q) / 2 + 1))
            feats.append(float(nu.item()))

        return np.array(feats, dtype=np.float32)

    # CPU fallback with NumPy
    I = img.astype(np.float64)
    h, w = I.shape
    Y, X = np.mgrid[:h, :w]

    m00 = I.sum()
    if m00 == 0:
        return np.zeros(8, dtype=np.float32)

    x_bar = (X * I).sum() / m00
    y_bar = (Y * I).sum() / m00

    feats = []
    for p, q in [(2, 0), (1, 1), (0, 2), (3, 0),
                 (2, 1), (1, 2), (0, 3), (2, 2)]:
        mu = (I * (X - x_bar) ** p * (Y - y_bar) ** q).sum()
        nu = mu / (m00 ** ((p + q) / 2 + 1))
        feats.append(nu)

    return np.array(feats, dtype=np.float32)