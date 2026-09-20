# dev/creator: tubakhxn

import cv2
import numpy as np

from config import HEATMAP_ALPHA

def build_density_map(points_xy, frame_w, frame_h, downscale=3, sigma=16, point_boost=3.0):
    small_w = max(1, frame_w // downscale)
    small_h = max(1, frame_h // downscale)
    accum = np.zeros((small_h, small_w), dtype=np.float32)

    for x, y in points_xy:
        sx = int(np.clip(x / downscale, 0, small_w - 1))
        sy = int(np.clip(y / downscale, 0, small_h - 1))
        accum[sy, sx] += point_boost

    k = max(3, int(round(sigma / downscale * 3)) | 1)
    blurred = cv2.GaussianBlur(accum, (k, k), sigmaX=sigma / downscale)

    if blurred.max() > 0:

        blurred = np.sqrt(blurred / blurred.max())

    full = cv2.resize(blurred, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    return full

def overlay_heatmap(frame_bgr, density_map, alpha=0.7):
    heat_u8 = np.uint8(np.clip(density_map, 0, 1) * 255)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heat_color = cv2.applyColorMap(heat_u8, colormap)

    mask = np.clip(density_map * 1.6, 0, 1).astype(np.float32)[..., None]
    blended = frame_bgr.astype(np.float32) * (1 - alpha * mask) +\
        heat_color.astype(np.float32) * (alpha * mask)
    return np.uint8(np.clip(blended, 0, 255))
