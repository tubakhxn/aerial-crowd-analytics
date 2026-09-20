# dev/creator: tubakhxn

import os
import time

import cv2
import numpy as np

from config import OUTPUT_DIR, YOLO_CONF_DEFAULT
from simulation.crowd_simulator import CrowdSimulator
from vision.detector import Detector
from vision.tracker import CentroidTracker
from vision.crowd_analytics import compute_density, filter_in_roi
from vision.pipeline import analyze_frame
from vision.video_processor import (
    make_blank_aerial_canvas, draw_detections, apply_heatmap_overlay, draw_hud,
    composite_bird_eye_inset, draw_glow_markers,
)

def _title_card(w, h, text_lines, duration_frames=15):
    frames = []
    canvas = np.full((h, w, 3), (8, 6, 4), dtype=np.uint8)
    y = h // 2 - (len(text_lines) * 16)
    for line in text_lines:
        size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (w - size[0]) // 2
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (238, 211, 34), 2, cv2.LINE_AA)
        y += 34
    for _ in range(duration_frames):
        frames.append(canvas.copy())
    return frames

def _open_writer(path, fps, w, h):
    for fourcc_str in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
    return None

def generate_demo_video(
    duration_seconds=6, fps=15, resolution=(960, 540),
    sim_config=None, roi_fraction=0.6, show_boxes=True, show_ids=True,
    show_heatmap=True, show_trails=True, intro_outro=True,
    progress_callback=None, source_frames=None, detector=None,
    conf_threshold=YOLO_CONF_DEFAULT,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    w, h = resolution
    total_frames = int(duration_seconds * fps)

    filename = f"drone_crowd_demo_{int(time.time())}.mp4"
    out_path = os.path.join(OUTPUT_DIR, filename)
    writer = _open_writer(out_path, fps, w, h)
    if writer is None:
        raise RuntimeError(
            "Could not open a video writer with any available codec "
            "(mp4v/avc1/XVID). Check OpenCV's codec support on this system."
        )

    sim_config = sim_config or {}
    simulator = CrowdSimulator(width=w, height=h, **sim_config)
    detector = detector or Detector()
    tracker = CentroidTracker()

    rx = w * (1 - roi_fraction) / 2
    ry = h * (1 - roi_fraction) / 2
    roi = (rx, ry, w - rx, h - ry)

    if intro_outro:
        for frame in _title_card(w, h, ["DRONE CROWD INTELLIGENCE"]):
            writer.write(frame)

    using_source = source_frames is not None
    modes_seen = set()

    for i in range(total_frames):
        if using_source:
            try:
                raw_frame = next(source_frames)
                raw_frame = cv2.resize(raw_frame, (w, h))
            except StopIteration:
                break
        else:
            raw_frame = make_blank_aerial_canvas(w, h)

        result = analyze_frame(
            frame_bgr=raw_frame, detector=detector, tracker=tracker, simulator=simulator,
            conf_threshold=conf_threshold, use_real_detection=using_source,
            advance_simulation=True, roi=roi, target_count=len(simulator.people),
        )
        points, detections, mode = result["points"], result["detections"], result["mode"]
        modes_seen.add(mode)

        frame = raw_frame
        if show_heatmap:
            frame = apply_heatmap_overlay(frame, points, enabled=True)

        frame = draw_detections(
            frame, detections, trails=tracker.trails if show_trails else None,
            show_boxes=show_boxes, show_ids=show_ids, show_conf=False,
            show_trails=show_trails, roi=roi,
        )
        frame = draw_glow_markers(frame, points, roi=roi, radius=4)
        frame = composite_bird_eye_inset(frame, points, w, h)

        in_roi = filter_in_roi(points, roi)
        density, label, _ = compute_density(len(in_roi), roi, w, h)

        hud = [
            f"DRONE CROWD INTELLIGENCE  [{mode.upper()}]",
            f"FRAME {i+1:04d}/{total_frames:04d}   FPS {fps}",
            f"TOTAL {len(points)}   ROI {len(in_roi)}   DENSITY {label.upper()} ({density:.2f})",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ]
        frame = draw_hud(frame, hud)

        writer.write(frame)
        if progress_callback:
            progress_callback((i + 1) / total_frames)

    if intro_outro:
        outro_lines = ["END OF DEMO CLIP"]
        if "synthetic" in modes_seen:
            outro_lines.append("SYNTHETIC DATA - NOT REAL SURVEILLANCE")
        elif "heuristic" in modes_seen:
            outro_lines.append("HEURISTIC CROWD ESTIMATE - NOT VERIFIED AI DETECTION")
        for frame in _title_card(w, h, outro_lines):
            writer.write(frame)

    writer.release()
    if modes_seen == {"real"}:
        overall_mode = "real"
    elif len(modes_seen) > 1:
        overall_mode = "mixed"
    elif modes_seen:
        overall_mode = next(iter(modes_seen))
    else:
        overall_mode = "synthetic"
    return {"path": out_path, "num_frames": total_frames, "mode": overall_mode}
