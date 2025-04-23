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


def extract_hu_features(img):
    """
    Returns eight scale-invariant central moments
    """
    mu = {}
    for (p, q) in [(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(2,2)]:
        mu[(p,q)] = central_moments(img, p, q)
    mu00 = mu[(0,0)] = central_moments(img, 0, 0)
    feats = []
    for (p,q) in [(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3),(2,2)]:
        feats.append(scale_invariant(mu[(p,q)], mu00, p, q))
    return np.array(feats, dtype=np.float32)