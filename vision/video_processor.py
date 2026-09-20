# dev/creator: tubakhxn

import time

import cv2
import numpy as np

from vision.heatmap import build_density_map, overlay_heatmap

CYAN = (238, 211, 34)
GREEN = (76, 217, 100)
ORANGE = (14, 129, 245)
RED = (68, 68, 239)
WHITE = (230, 240, 246)

def make_blank_aerial_canvas(w, h, seed=7):
    rng = np.random.default_rng(seed)
    canvas = np.full((h, w, 3), (18, 14, 10), dtype=np.uint8)
    block_w, block_h = 70, 46
    for by in range(0, h, block_h):
        for bx in range(0, w, block_w):
            shade = int(rng.integers(28, 55))
            cv2.rectangle(canvas, (bx + 3, by + 3), (bx + block_w - 3, by + block_h - 3),
                          (shade, shade - 4, shade - 8), -1)
    return canvas

def draw_detections(frame, detections, tracker_positions=None, trails=None,
                     show_boxes=True, show_ids=True, show_conf=True, show_trails=False,
                     roi=None):
    out = frame.copy()

    if roi is not None:
        x1, y1, x2, y2 = [int(v) for v in roi]
        cv2.rectangle(out, (x1, y1), (x2, y2), CYAN, 2)
        cv2.putText(out, "ROI", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)

    if show_trails and trails:
        for tid, trail in trails.items():
            pts = np.array(trail, dtype=np.int32)
            for i in range(1, len(pts)):
                cv2.line(out, tuple(pts[i - 1]), tuple(pts[i]), (255, 180, 40), 1, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        in_roi = False
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            cx, cy = det["cx"], det["cy"]
            in_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        color = GREEN if in_roi else (WHITE if roi is None else (120, 120, 120))

        if show_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

        label_parts = []
        if show_ids and "track_id" in det:
            label_parts.append(f"ID{det['track_id']}")
        if show_conf:
            label_parts.append(f"{det['conf']:.2f}")
        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(out, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, color, 1, cv2.LINE_AA)

    return out

def apply_heatmap_overlay(frame, points_xy, enabled=True):
    if not enabled or not points_xy:
        return frame
    dmap = build_density_map(points_xy, frame.shape[1], frame.shape[0])
    return overlay_heatmap(frame, dmap)

def draw_glow_markers(frame_bgr, points_xy, roi=None, radius=4, core_color=(60, 255, 120),
                       ring_color=(220, 255, 220), glow_strength=0.85, draw_ring=False):
    if not points_xy:
        return frame_bgr

    glow_layer = np.zeros_like(frame_bgr, dtype=np.float32)
    out = frame_bgr.copy()

    for x, y in points_xy:
        cx, cy = int(round(x)), int(round(y))
        in_roi = True
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            in_roi = rx1 <= x <= rx2 and ry1 <= y <= ry2
        color = core_color if in_roi else (60, 180, 255)

        cv2.circle(glow_layer, (cx, cy), radius * 2, color, -1, cv2.LINE_AA)

    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=radius * 1.1)
    out = np.uint8(np.clip(out.astype(np.float32) + glow_layer * glow_strength, 0, 255))

    for x, y in points_xy:
        cx, cy = int(round(x)), int(round(y))
        in_roi = True
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            in_roi = rx1 <= x <= rx2 and ry1 <= y <= ry2
        color = core_color if in_roi else (60, 180, 255)
        if draw_ring:
            cv2.circle(out, (cx, cy), radius, color, 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), max(1, radius // 2), ring_color, -1, cv2.LINE_AA)

    return out

def draw_hud(frame, hud_lines, corner="top-left"):
    out = frame.copy()
    x, y = 10, 20
    for line in hud_lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)
        y += 18
    return out

def composite_bird_eye_inset(frame, points, frame_w, frame_h, inset_w=220, inset_h=140, margin=12):
    from vision.heatmap import build_density_map, overlay_heatmap

    panel = np.full((inset_h, inset_w, 3), (10, 8, 6), dtype=np.uint8)
    if points:
        sx, sy = inset_w / frame_w, inset_h / frame_h
        scaled = [(x * sx, y * sy) for (x, y) in points]
        dmap = build_density_map(scaled, inset_w, inset_h, downscale=2, sigma=6)
        panel = overlay_heatmap(panel, dmap, alpha=0.9)
        for x, y in scaled:
            cv2.circle(panel, (int(x), int(y)), 1, (140, 255, 140), -1, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (inset_w - 1, inset_h - 1), CYAN, 1)
    cv2.putText(panel, "BIRD'S EYE VIEW", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1, cv2.LINE_AA)

    out = frame.copy()
    x0 = frame_w - inset_w - margin
    y0 = margin
    out[y0:y0 + inset_h, x0:x0 + inset_w] = panel
    return out

def timestamp_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")
