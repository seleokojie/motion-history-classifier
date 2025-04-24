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
    Compute central moment mu_{pq}
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
    return mu_pq / (mu_00 ** ((p + q) / 2 + 1)) if mu_00 > 0 else 0.0


def extract_hu_features(img: np.ndarray) -> np.ndarray:
    """
    Compute the 8 scale‐invariant central moments (Hu style),
    doing all the heavy lifting on GPU via CuPy and then
    returning a 8‐dim NumPy array.
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