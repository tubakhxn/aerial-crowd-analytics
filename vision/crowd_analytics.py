# dev/creator: tubakhxn

import numpy as np

from config import DENSITY_THRESHOLDS, DENSITY_COLORS

def point_in_roi(x, y, roi):
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2

def filter_in_roi(points_xy, roi):
    return [(x, y) for (x, y) in points_xy if point_in_roi(x, y, roi)]

def roi_area_norm(roi, frame_w, frame_h):
    x1, y1, x2, y2 = roi
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    frame_area = max(1.0, frame_w * frame_h)
    return area / frame_area

def compute_density(count_in_roi, roi, frame_w, frame_h, reference_capacity=400):
    area_frac = max(roi_area_norm(roi, frame_w, frame_h), 0.01)
    capacity = reference_capacity * area_frac
    density = count_in_roi / max(capacity, 1.0)

    if density < DENSITY_THRESHOLDS["low"]:
        label = "low"
    elif density < DENSITY_THRESHOLDS["moderate"]:
        label = "moderate"
    elif density < DENSITY_THRESHOLDS["high"]:
        label = "high"
    else:
        label = "critical"

    return density, label, DENSITY_COLORS[label]

def average_speed(tracker, ids):
    speeds = []
    for oid in ids:
        trail = tracker.get_trail(oid)
        if len(trail) >= 2:
            (x1, y1), (x2, y2) = trail[-2], trail[-1]
            speeds.append(float(np.hypot(x2 - x1, y2 - y1)))
    return float(np.mean(speeds)) if speeds else 0.0

def flow_direction(tracker, ids):
    vecs = []
    for oid in ids:
        trail = tracker.get_trail(oid)
        if len(trail) >= 2:
            (x1, y1), (x2, y2) = trail[-2], trail[-1]
            vecs.append((x2 - x1, y2 - y1))
    if not vecs:
        return 0.0, 0.0
    vx = float(np.mean([v[0] for v in vecs]))
    vy = float(np.mean([v[1] for v in vecs]))
    angle = float(np.degrees(np.arctan2(vy, vx))) % 360
    magnitude = float(np.hypot(vx, vy))
    return angle, magnitude

def high_density_cells(points_xy, frame_w, frame_h, grid=(8, 8), top_k=3):
    if not points_xy:
        return []
    gx, gy = grid
    counts = np.zeros((gy, gx))
    cw, ch = frame_w / gx, frame_h / gy
    for x, y in points_xy:
        cx = min(int(x // cw), gx - 1)
        cy = min(int(y // ch), gy - 1)
        counts[cy, cx] += 1

    flat_idx = np.argsort(counts, axis=None)[::-1][:top_k]
    cells = []
    for idx in flat_idx:
        cy, cx = np.unravel_index(idx, counts.shape)
        if counts[cy, cx] <= 0:
            continue
        cells.append({
            "x": (cx + 0.5) * cw, "y": (cy + 0.5) * ch,
            "count": int(counts[cy, cx]),
        })
    return cells
