# dev/creator: tubakhxn

import streamlit as st

from config import YOLO_MODEL_NAME, YOLO_CONF_DEFAULT, YOLO_PERSON_CLASS_ID

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_NAME)
        return model, True
    except Exception:
        return None, False

class Detector:

    def __init__(self, confidence=YOLO_CONF_DEFAULT):
        self.confidence = confidence
        self.model, self.is_real = load_yolo_model()

    @property
    def mode(self):
        return "real" if self.is_real and self.model is not None else "synthetic"

    def detect_frame(self, frame_bgr):
        if not self.is_real or self.model is None:
            raise RuntimeError("Real detection model not available")

        results = self.model.predict(
            frame_bgr, conf=self.confidence, classes=[YOLO_PERSON_CLASS_ID], verbose=False
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({"box": (x1, y1, x2, y2), "conf": conf, "cx": cx, "cy": cy})
        return detections

    @staticmethod
    def synthetic_boxes_from_points(people, box_size=18):
        detections = []
        for p in people:
            x, y = p["x"], p["y"]
            half = box_size / 2
            conf = 0.72 + 0.25 * ((p["id"] % 7) / 7.0)
            detections.append({
                "box": (x - half, y - half, x + half, y + half),
                "conf": round(min(conf, 0.98), 2),
                "cx": x, "cy": y,
                "track_id": p["id"],
            })
        return detections
