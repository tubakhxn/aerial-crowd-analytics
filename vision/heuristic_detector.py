# dev/creator: tubakhxn

import cv2
import numpy as np

def _texture_color_mask(frame_bgr, downscale=3):
    small = cv2.resize(frame_bgr, (frame_bgr.shape[1] // downscale, frame_bgr.shape[0] // downscale))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    def local_std(img, k=9):
        mean = cv2.blur(img, (k, k))
        mean_sq = cv2.blur(img * img, (k, k))
        return np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))

    texture = local_std(gray)
    color_var = local_std(sat)
    sat_level = cv2.blur(sat, (9, 9))

    texture_n = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    color_n = cv2.normalize(color_var, None, 0, 255, cv2.NORM_MINMAX)
    sat_n = cv2.normalize(sat_level, None, 0, 255, cv2.NORM_MINMAX)

    combined = np.uint8(np.clip(0.22 * texture_n + 0.20 * color_n + 0.58 * sat_n, 0, 255))
    return combined, gray.astype(np.uint8), downscale

def estimate_crowd_points(frame_bgr, target_count, roi=None, seed=None):
    if target_count <= 0:
        return []

    combined, gray_small, downscale = _texture_color_mask(frame_bgr)
    h_s, w_s = combined.shape

    thresh_val = max(10, int(np.percentile(combined, 68)))
    mask = (combined >= thresh_val).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    corners = cv2.goodFeaturesToTrack(
        gray_small, maxCorners=max(target_count * 4, 60), qualityLevel=0.008,
        minDistance=5, mask=mask,
    )

    points = []
    if corners is not None:
        for c in corners[:, 0, :]:
            x, y = float(c[0]) * downscale, float(c[1]) * downscale
            points.append((x, y))

    if not points:
        return []

    rng = np.random.default_rng(seed)

    if roi is not None and len(points) > target_count:
        rx1, ry1, rx2, ry2 = roi
        in_roi = [p for p in points if rx1 <= p[0] <= rx2 and ry1 <= p[1] <= ry2]
        out_roi = [p for p in points if p not in in_roi]
        n_in = min(len(in_roi), int(target_count * 0.75))
        n_out = min(len(out_roi), target_count - n_in)
        chosen = []
        if in_roi:
            idx = rng.choice(len(in_roi), n_in, replace=False)
            chosen += [in_roi[i] for i in idx]
        if out_roi:
            idx = rng.choice(len(out_roi), n_out, replace=False)
            chosen += [out_roi[i] for i in idx]
        points = chosen
    elif len(points) > target_count:
        idx = rng.choice(len(points), target_count, replace=False)
        points = [points[i] for i in idx]

    return points
