# dev/creator: tubakhxn

DEFAULT_FRAME_WIDTH = 960
DEFAULT_FRAME_HEIGHT = 540
DEFAULT_FRAME_SKIP = 1
MODEL_CACHE_KEY = "yolo_model_v1"

DENSITY_THRESHOLDS = {
    "low": 0.15,
    "moderate": 0.35,
    "high": 0.60,
    "critical": 1.0,
}

DENSITY_COLORS = {
    "low": "#00e676",
    "moderate": "#ffd600",
    "high": "#ff9100",
    "critical": "#ff1744",
}

SIM_DEFAULTS = {
    "num_people": 220,
    "density_bias": 0.5,
    "movement_speed": 1.2,
    "roi_fraction": 0.6,
    "duration_frames": 300,
    "num_clusters": 4,
}

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF_DEFAULT = 0.35
YOLO_PERSON_CLASS_ID = 0

TRACKER_MAX_DISAPPEARED = 15
TRACKER_MAX_DISTANCE = 80

HEATMAP_BLUR_KSIZE = 31
HEATMAP_ALPHA = 0.55

EXPORT_DEFAULT_FPS = 15
EXPORT_DEFAULT_RESOLUTION = (960, 540)
EXPORT_PREVIEW_SECONDS = 6
OUTPUT_DIR = "outputs"

THEME = {
    "bg": "#05080f",
    "panel_bg": "#0a0f1a",
    "border": "#1c2a3a",
    "cyan": "#22d3ee",
    "blue": "#3b82f6",
    "green": "#22c55e",
    "orange": "#f59e0b",
    "red": "#ef4444",
    "text": "#d6e4f0",
    "muted": "#6b7f95",
}
