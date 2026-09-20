# dev/creator: tubakhxn

import time
import tempfile
from collections import deque

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import config
from simulation.crowd_simulator import CrowdSimulator
from vision.detector import Detector
from vision.tracker import CentroidTracker
from vision.pipeline import analyze_frame
from vision.crowd_analytics import (
    filter_in_roi, compute_density, average_speed, flow_direction, high_density_cells,
)
from vision.video_processor import make_blank_aerial_canvas, draw_detections, apply_heatmap_overlay
from visualization.bird_eye_view import build_bird_eye_figure
from visualization.crowd_3d import build_3d_density_figure
from export.video_generator import generate_demo_video

st.set_page_config(page_title="Drone Crowd Intelligence", layout="wide", initial_sidebar_state="expanded")

T = config.THEME

st.markdown(f"""
<style>
.stApp {{ background-color: {T['bg']}; color: {T['text']}; }}
section[data-testid="stSidebar"] {{ background-color: {T['panel_bg']}; border-right: 1px solid {T['border']}; }}
.hud-bar {{
    background: {T['panel_bg']}; border: 1px solid {T['border']}; border-radius: 4px;
    padding: 10px 16px; margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 26px;
    align-items: center; font-family: 'Courier New', monospace;
}}
.hud-title {{ color: {T['cyan']}; font-weight: 700; letter-spacing: 1px; font-size: 15px; }}
.hud-item {{ font-size: 12px; color: {T['muted']}; }}
.hud-val {{ color: {T['text']}; font-weight: 700; font-size: 14px; }}
.status-live {{ color: {T['red']}; font-weight: 700; }}
.status-sim {{ color: {T['green']}; font-weight: 700; }}
.panel-title {{
    color: {T['cyan']}; font-family: 'Courier New', monospace; font-size: 12px;
    letter-spacing: 1px; border-bottom: 1px solid {T['border']}; padding-bottom: 4px; margin-bottom: 6px;
}}
.metric-box {{
    background: {T['panel_bg']}; border: 1px solid {T['border']}; border-radius: 4px;
    padding: 8px 10px; text-align: center;
}}
.metric-label {{ font-size: 10px; color: {T['muted']}; letter-spacing: 1px; }}
.metric-value {{ font-size: 20px; font-weight: 700; }}
.alert-row {{ font-family: 'Courier New', monospace; font-size: 11px; padding: 2px 0; }}
</style>
""", unsafe_allow_html=True)

def init_state():
    ss = st.session_state
    ss.setdefault("running", False)
    ss.setdefault("frame_idx", 0)
    ss.setdefault("width", config.DEFAULT_FRAME_WIDTH)
    ss.setdefault("height", config.DEFAULT_FRAME_HEIGHT)
    ss.setdefault("simulator", CrowdSimulator(
        width=ss.get("width", config.DEFAULT_FRAME_WIDTH),
        height=ss.get("height", config.DEFAULT_FRAME_HEIGHT),
        num_people=config.SIM_DEFAULTS["num_people"],
        density_bias=config.SIM_DEFAULTS["density_bias"],
        movement_speed=config.SIM_DEFAULTS["movement_speed"],
        num_clusters=config.SIM_DEFAULTS["num_clusters"],
    ))
    ss.setdefault("tracker", CentroidTracker(
        max_disappeared=config.TRACKER_MAX_DISAPPEARED, max_distance=config.TRACKER_MAX_DISTANCE))
    ss.setdefault("detector", Detector())
    ss.setdefault("count_history", deque(maxlen=120))
    ss.setdefault("density_history", deque(maxlen=120))
    ss.setdefault("conf_history", deque(maxlen=120))
    ss.setdefault("alert_log", deque(maxlen=50))
    ss.setdefault("video_capture", None)
    ss.setdefault("video_source_name", "Synthetic Aerial Scene")
    ss.setdefault("last_generated_video", None)
    ss.setdefault("start_time", time.time())

init_state()
ss = st.session_state

with st.sidebar:
    st.markdown("### ⚙️ CONTROL PANEL")

    st.markdown("**Playback**")
    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Start", use_container_width=True):
        ss.running = True
    if c2.button("⏸ Pause", use_container_width=True):
        ss.running = False
    if c3.button("⟲ Reset", use_container_width=True):
        ss.simulator.reset()
        ss.tracker = CentroidTracker(max_disappeared=config.TRACKER_MAX_DISAPPEARED,
                                      max_distance=config.TRACKER_MAX_DISTANCE)
        ss.frame_idx = 0
        ss.count_history.clear(); ss.density_history.clear(); ss.conf_history.clear()
        ss.alert_log.clear()
        ss.running = False

    playback_speed = st.slider("Playback speed", 0.25, 3.0, 1.0, 0.25)

    st.markdown("---")
    st.markdown("**Video Source**")
    src_mode = st.radio("Feed", ["Synthetic sample (bundled)", "Upload video"], label_visibility="collapsed")
    if src_mode == "Upload video":
        uploaded = st.file_uploader("Upload footage (mp4/mov/avi)", type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None and ss.video_source_name != uploaded.name:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            ss.video_capture = cv2.VideoCapture(tfile.name)
            ss.video_source_name = uploaded.name
    else:
        ss.video_capture = None
        ss.video_source_name = "Synthetic Aerial Scene"

    st.markdown("---")
    st.markdown("**Simulation Parameters**")
    n_people = st.slider("Simulated people", 20, 600, config.SIM_DEFAULTS["num_people"], 10)
    density_bias = st.slider("Clustering (crowd tightness)", 0.0, 1.0, config.SIM_DEFAULTS["density_bias"], 0.05)
    move_speed = st.slider("Movement speed", 0.1, 4.0, config.SIM_DEFAULTS["movement_speed"], 0.1)
    ss.simulator.set_population(n_people)
    ss.simulator.density_bias = density_bias
    ss.simulator.movement_speed = move_speed

    st.markdown("---")
    st.markdown("**Region of Interest**")
    roi_frac = st.slider("ROI size (% of frame)", 0.2, 1.0, config.SIM_DEFAULTS["roi_fraction"], 0.05)
    roi_offset_x = st.slider("ROI horizontal offset", -0.3, 0.3, 0.0, 0.02)
    roi_offset_y = st.slider("ROI vertical offset", -0.3, 0.3, 0.0, 0.02)
    show_roi = st.checkbox("Show ROI overlay", value=True)

    st.markdown("---")
    st.markdown("**Detection & Overlays**")
    enable_detections = st.checkbox("Enable detections", value=True)
    show_boxes = st.checkbox("Show bounding boxes", value=True)
    show_ids = st.checkbox("Show tracking IDs", value=True)
    show_conf = st.checkbox("Show confidence labels", value=True)
    show_heatmap = st.checkbox("Enable density heatmap", value=True)
    show_trails = st.checkbox("Show movement trails", value=False)
    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.95, config.YOLO_CONF_DEFAULT, 0.05)

    st.markdown("---")
    st.markdown("**Alert Thresholds** (demo only, not safety-certified)")
    thresh_high = st.slider("High density trigger", 0.3, 0.9, config.DENSITY_THRESHOLDS["high"], 0.05)
    thresh_critical = st.slider("Critical density trigger", 0.6, 1.5, config.DENSITY_THRESHOLDS["critical"], 0.05)
    config.DENSITY_THRESHOLDS["high"] = thresh_high
    config.DENSITY_THRESHOLDS["critical"] = thresh_critical

    st.markdown("---")
    st.markdown("**🎬 Export**")
    clip_seconds = st.slider("Demo clip duration (s)", 2, 15, 6)
    export_fps = st.slider("Export FPS", 5, 30, 15)
    generate_clicked = st.button("🎬 Generate Demo Video", use_container_width=True)

W, H = ss.width, ss.height
rw, rh = W * roi_frac, H * roi_frac
cx, cy = W / 2 + roi_offset_x * W, H / 2 + roi_offset_y * H
roi = (max(0, cx - rw / 2), max(0, cy - rh / 2), min(W, cx + rw / 2), min(H, cy + rh / 2))

def read_source_frame():
    if ss.video_capture is not None:
        ok, frame = ss.video_capture.read()
        if not ok:
            ss.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = ss.video_capture.read()
        if ok:
            return cv2.resize(frame, (W, H))
    return make_blank_aerial_canvas(W, H)

if ss.running:
    ss.frame_idx += 1

frame = read_source_frame()
use_real_detection = enable_detections and ss.video_capture is not None

result = analyze_frame(
    frame_bgr=frame, detector=ss.detector, tracker=ss.tracker, simulator=ss.simulator,
    conf_threshold=conf_threshold, use_real_detection=use_real_detection,
    advance_simulation=ss.running, roi=roi, target_count=n_people,
)
points = result["points"] if enable_detections else []
detections = result["detections"] if enable_detections else []
active_mode = result["mode"]

if show_heatmap:
    frame = apply_heatmap_overlay(frame, points, enabled=True)
frame = draw_detections(
    frame, detections, trails=ss.tracker.trails if show_trails else None,
    show_boxes=show_boxes, show_ids=show_ids, show_conf=show_conf,
    show_trails=show_trails, roi=roi if show_roi else None,
)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

in_roi_points = filter_in_roi(points, roi)
density_val, density_label, density_color = compute_density(len(in_roi_points), roi, W, H)
ids_in_roi = [d.get("track_id") for d in detections if d.get("track_id") is not None]
avg_speed = average_speed(ss.tracker, ids_in_roi) if ss.tracker.trails else 0.0
angle, mag = flow_direction(ss.tracker, ids_in_roi) if ss.tracker.trails else (0.0, 0.0)
hot_cells = high_density_cells(points, W, H)
mean_conf = float(np.mean([d["conf"] for d in detections])) if detections else 0.0
elapsed = time.time() - ss.start_time
fps_est = (ss.frame_idx / elapsed) if elapsed > 0 else 0.0

if ss.running:
    ss.count_history.append(len(points))
    ss.density_history.append(density_val)
    ss.conf_history.append(mean_conf)
    if density_label in ("high", "critical"):
        ss.alert_log.appendleft(
            f"[{time.strftime('%H:%M:%S')}] {density_label.upper()} density in ROI "
            f"({len(in_roi_points)} people, {density_val:.2f})"
        )

mode_tag = {
    "real": "LIVE (REAL DETECTION)",
    "heuristic": "LIVE (CONTENT-AWARE ESTIMATE)",
    "synthetic": "LIVE (SIMULATION)",
}.get(active_mode, "LIVE")
status_html = f'<span class="status-live">● {mode_tag}</span>' if ss.running else '<span class="status-sim">■ PAUSED</span>'
alerts_count = sum(1 for _ in list(ss.alert_log)[:10])
st.markdown(f"""
<div class="hud-bar">
  <div class="hud-title">🚁 DRONE CROWD INTELLIGENCE</div>
  {status_html}
  <div class="hud-item">DRONE ID <span class="hud-val">DX-07</span></div>
  <div class="hud-item">FRAME <span class="hud-val">{ss.frame_idx:05d}</span></div>
  <div class="hud-item">FPS <span class="hud-val">{fps_est:0.1f}</span></div>
  <div class="hud-item">TOTAL PEOPLE <span class="hud-val">{len(points)}</span></div>
  <div class="hud-item">IN ROI <span class="hud-val">{len(in_roi_points)}</span></div>
  <div class="hud-item">DENSITY <span class="hud-val" style="color:{density_color}">{density_label.upper()} {density_val:.2f}</span></div>
  <div class="hud-item">ALERTS <span class="hud-val" style="color:{T['red']}">{len(ss.alert_log)}</span></div>
  <div class="hud-item">SOURCE <span class="hud-val">{ss.video_source_name}</span></div>
</div>
""", unsafe_allow_html=True)

if active_mode == "real":
    st.caption("⚠️ Educational research demo. Boxes/counts below come from a REAL YOLO person-detection "
               "model running on your uploaded footage. Not a certified public-safety system.")
elif active_mode == "heuristic":
    st.caption("⚠️ Educational research demo. No real AI model is loaded, so markers are placed using "
               "content-aware image analysis of your actual footage (texture + color) — NOT verified "
               "AI person detection. Install `ultralytics` for real detection. Not a certified "
               "public-safety system.")
else:
    reason = ""
    if ss.video_capture is not None and ss.detector.mode != "real":
        reason = (" (YOLO/ultralytics isn't loaded — install it with `pip install ultralytics` "
                   "for real detection on your footage.)")
    st.caption(f"⚠️ Educational research demo. Crowd counts/density here are SIMULATED{reason} "
               "Not a certified public-safety system.")

left, right = st.columns([1.6, 1])

with left:
    st.markdown('<div class="panel-title">MAIN FEED — AERIAL VIEW</div>', unsafe_allow_html=True)
    st.image(frame_rgb, use_container_width=True)

with right:
    st.markdown('<div class="panel-title">BIRD\'S-EYE DENSITY VIEW</div>', unsafe_allow_html=True)
    bev_fig = build_bird_eye_figure(points, W, H, roi=roi if show_roi else None,
                                     flow_angle=angle, flow_mag=mag)
    st.plotly_chart(bev_fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="panel-title">3D CROWD DENSITY RECONSTRUCTION</div>', unsafe_allow_html=True)
c3d1, c3d2 = st.columns([3, 1])
with c3d2:
    view_mode = st.radio("3D mode", ["surface", "points"], horizontal=False, label_visibility="collapsed")
with c3d1:
    fig3d = build_3d_density_figure(points, W, H, mode=view_mode)
    st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="panel-title">ANALYTICS</div>', unsafe_allow_html=True)
a1, a2, a3, a4, a5 = st.columns(5)

with a1:
    st.markdown(f'<div class="metric-box"><div class="metric-label">AVG SPEED (px/frame)</div>'
                f'<div class="metric-value">{avg_speed:0.2f}</div></div>', unsafe_allow_html=True)
with a2:
    st.markdown(f'<div class="metric-box"><div class="metric-label">FLOW ANGLE</div>'
                f'<div class="metric-value">{angle:0.0f}°</div></div>', unsafe_allow_html=True)
with a3:
    st.markdown(f'<div class="metric-box"><div class="metric-label">HOTSPOTS</div>'
                f'<div class="metric-value">{len(hot_cells)}</div></div>', unsafe_allow_html=True)
with a4:
    st.markdown(f'<div class="metric-box"><div class="metric-label">MEAN CONFIDENCE</div>'
                f'<div class="metric-value">{mean_conf:0.2f}</div></div>', unsafe_allow_html=True)
with a5:
    st.markdown(f'<div class="metric-box"><div class="metric-label">DETECTOR MODE</div>'
                f'<div class="metric-value" style="font-size:14px">{active_mode.upper()}</div></div>',
                unsafe_allow_html=True)

st.write("")
b1, b2 = st.columns(2)
with b1:
    fig_count = go.Figure()
    fig_count.add_trace(go.Scatter(y=list(ss.count_history), mode="lines",
                                    line=dict(color=T["cyan"], width=2), fill="tozeroy"))
    fig_count.update_layout(title="Crowd count over time", height=200,
                             paper_bgcolor=T["panel_bg"], plot_bgcolor=T["panel_bg"],
                             font=dict(color=T["muted"], size=10),
                             margin=dict(l=4, r=4, t=30, b=4))
    st.plotly_chart(fig_count, use_container_width=True, config={"displayModeBar": False})
with b2:
    fig_density = go.Figure()
    fig_density.add_trace(go.Scatter(y=list(ss.density_history), mode="lines",
                                      line=dict(color=T["orange"], width=2), fill="tozeroy"))
    fig_density.update_layout(title="Density over time", height=200,
                               paper_bgcolor=T["panel_bg"], plot_bgcolor=T["panel_bg"],
                               font=dict(color=T["muted"], size=10),
                               margin=dict(l=4, r=4, t=30, b=4))
    st.plotly_chart(fig_density, use_container_width=True, config={"displayModeBar": False})

b3, b4 = st.columns(2)
with b3:
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Scatter(y=list(ss.conf_history), mode="lines",
                                   line=dict(color=T["green"], width=2)))
    fig_conf.update_layout(title="Detection confidence", height=180,
                            paper_bgcolor=T["panel_bg"], plot_bgcolor=T["panel_bg"],
                            font=dict(color=T["muted"], size=10),
                            margin=dict(l=4, r=4, t=30, b=4))
    st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})
with b4:
    st.markdown('<div class="panel-title" style="border:none;">ALERT TIMELINE</div>', unsafe_allow_html=True)
    if ss.alert_log:
        for line in list(ss.alert_log)[:8]:
            st.markdown(f'<div class="alert-row" style="color:{T["red"]}">⚠ {line}</div>', unsafe_allow_html=True)
    else:
        st.caption("No alerts yet.")

def _video_frame_generator(cap, total_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    yielded = 0
    while yielded < total_frames:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        yield frame
        yielded += 1

if generate_clicked:
    progress = st.progress(0, text="Rendering demo video…")

    def _cb(frac):
        progress.progress(min(1.0, frac), text=f"Rendering demo video… {int(frac*100)}%")

    total_export_frames = clip_seconds * export_fps
    source_gen = (
        _video_frame_generator(ss.video_capture, total_export_frames)
        if ss.video_capture is not None else None
    )

    export_result = generate_demo_video(
        duration_seconds=clip_seconds, fps=export_fps, resolution=(W, H),
        sim_config=dict(num_people=n_people, density_bias=density_bias,
                         movement_speed=move_speed, num_clusters=config.SIM_DEFAULTS["num_clusters"]),
        roi_fraction=roi_frac, show_boxes=show_boxes, show_ids=show_ids,
        show_heatmap=show_heatmap, show_trails=show_trails, intro_outro=True,
        progress_callback=_cb, source_frames=source_gen, detector=ss.detector,
        conf_threshold=conf_threshold,
    )
    progress.progress(1.0, text="Done.")
    ss.last_generated_video = export_result["path"]
    st.success(f"Demo video generated ({export_result['num_frames']} frames, "
               f"mode: {export_result['mode']}, source: {ss.video_source_name}).")

if ss.last_generated_video:
    st.markdown('<div class="panel-title">GENERATED DEMO VIDEO</div>', unsafe_allow_html=True)
    with open(ss.last_generated_video, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)
    st.download_button("⬇ Download MP4", data=video_bytes,
                        file_name=ss.last_generated_video.split("/")[-1], mime="video/mp4")

if ss.running:
    time.sleep(max(0.02, 0.12 / playback_speed))
    st.rerun()
