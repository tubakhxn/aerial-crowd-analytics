# dev/creator: tubakhxn

from vision.heuristic_detector import estimate_crowd_points

def analyze_frame(frame_bgr, detector, tracker, simulator, conf_threshold,
                   use_real_detection, advance_simulation=True, roi=None,
                   target_count=None):

    if use_real_detection and detector.mode == "real" and frame_bgr is not None:
        try:
            raw_detections = detector.detect_frame(frame_bgr)
        except Exception:
            raw_detections = None

        if raw_detections is not None:
            raw_detections = [d for d in raw_detections if d["conf"] >= conf_threshold]
            centroids = [(d["cx"], d["cy"]) for d in raw_detections]
            tracked = tracker.update(centroids) if advance_simulation else dict(tracker.objects)
            pos_to_id = {v: k for k, v in tracked.items()}
            for d in raw_detections:
                d["track_id"] = pos_to_id.get((d["cx"], d["cy"]))
            return {"points": centroids, "detections": raw_detections, "mode": "real"}

    if use_real_detection and frame_bgr is not None:
        h, w = frame_bgr.shape[:2]
        count = target_count or max(20, len(simulator.people))
        points = estimate_crowd_points(frame_bgr, target_count=count, roi=roi)
        if points:
            tracked = tracker.update(points) if advance_simulation else dict(tracker.objects)
            pos_to_id = {v: k for k, v in tracked.items()}
            detections = []
            for (x, y) in points:
                half = 9
                detections.append({
                    "box": (x - half, y - half, x + half, y + half),
                    "conf": 0.55, "cx": x, "cy": y,
                    "track_id": pos_to_id.get((x, y)),
                })
            detections = [d for d in detections if d["conf"] >= conf_threshold]
            return {"points": points, "detections": detections, "mode": "heuristic"}

    people = simulator.step() if advance_simulation else simulator.get_people()
    points = [(p["x"], p["y"]) for p in people]
    centroids = points[:]
    tracker.update(centroids) if advance_simulation else dict(tracker.objects)
    detections = detector.synthetic_boxes_from_points(people)
    detections = [d for d in detections if d["conf"] >= conf_threshold]
    return {"points": points, "detections": detections, "mode": "synthetic"}
