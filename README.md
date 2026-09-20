# Drone Crowd Intelligence System

## Dev/Creator: tubakhxn

This is a locally runnable computer vision research dashboard that simulates
drone based crowd monitoring. It combines a live video dashboard, crowd
density analytics, a bird's eye density view, a 3D crowd reconstruction
panel, and an annotated MP4 export pipeline, built with Python, Streamlit,
OpenCV, and Plotly.

This is an educational and research demo, not a certified public safety or
crowd control system. Density thresholds used in the dashboard are
illustrative defaults, not validated safety standards. Unless a real object
detection model is active, all detections and counts are either fully
simulated or estimated using basic image analysis, and this is always
labeled in the interface.
<img width="2206" height="1418" alt="image" src="https://github.com/user-attachments/assets/37268a83-2380-4afa-b72c-2cda3cd1bc69" />
<img width="2196" height="1486" alt="image" src="https://github.com/user-attachments/assets/72d736ef-32f2-4ae5-8e4b-5548150ddeab" />


## What the project does

- Simulates or analyzes an aerial crowd scene and estimates people count,
  density, and movement
- Optionally runs real person detection using YOLO (via Ultralytics) if the
  model is installed and its weights can be downloaded
- Falls back to a content aware heuristic estimator that analyzes texture
  and color in an uploaded video to approximate where a crowd is located,
  when no real detection model is available
- Falls back further to a fully synthetic simulated crowd if no video is
  provided at all, so the dashboard always has something to display
- Tracks detected points across frames using a lightweight centroid
  tracker and assigns IDs
- Computes region of interest based analytics: count, density
  classification, average movement speed, and flow direction
- Renders a bird's eye density view and a 3D density reconstruction using
  Plotly
- Exports a short annotated MP4 clip with heatmap, region of interest
  overlay, and a bird's eye inset panel

## Background concepts

These Wikipedia articles give useful background on the ideas this project
is built around:

- Computer vision: https://en.wikipedia.org/wiki/Computer_vision
- Object detection: https://en.wikipedia.org/wiki/Object_detection
- You Only Look Once (YOLO): https://en.wikipedia.org/wiki/You_Only_Look_Once
- Crowd simulation: https://en.wikipedia.org/wiki/Crowd_simulation
- Crowd counting: https://en.wikipedia.org/wiki/Crowd_counting
- Heat map: https://en.wikipedia.org/wiki/Heat_map
- OpenCV: https://en.wikipedia.org/wiki/OpenCV
- Video tracking: https://en.wikipedia.org/wiki/Video_tracking
- Unmanned aerial vehicle: https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle
- Aerial photography: https://en.wikipedia.org/wiki/Aerial_photography

## Requirements

- Python 3.10 or newer
- pip

## Installation

Clone or download this repository, then from inside the project folder:

```
pip install -r requirements.txt
```

The `ultralytics` package is listed as an optional dependency for real YOLO
based detection. The app works fully without it. If it is not installed, or
its model weights cannot be downloaded, the app automatically falls back to
the heuristic or simulated modes described above.

## Running it locally

From inside the project folder:

```
streamlit run app.py
```

or, if the `streamlit` command is not on your PATH:

```
python -m streamlit run app.py
```

Streamlit will start a local web server and print a URL, typically:

```
http://localhost:8501
```

Open that URL in your browser. The dashboard runs entirely on your own
machine; nothing is uploaded to a remote server.

## Forking and modifying this project

1. Copy this project folder to your own machine or your own repository.
2. Make your changes inside the relevant module:
   - `app.py` for the dashboard layout and controls
   - `config.py` for constants, thresholds, and the color theme
   - `vision/` for detection, tracking, analytics, and heatmap logic
   - `simulation/` for the synthetic crowd generator
   - `visualization/` for the bird's eye and 3D Plotly views
   - `export/` for the MP4 export pipeline
3. Re-run `pip install -r requirements.txt` if you add new dependencies.
4. Run `streamlit run app.py` to test your changes locally.
5. If you intend to publish your fork, keep the educational and research
   disclaimers in the interface, since the underlying detections can be
   simulated or estimated rather than verified in every mode.

## Project structure

```
drone-crowd-intelligence/
    app.py
    config.py
    requirements.txt
    vision/
        detector.py
        tracker.py
        crowd_analytics.py
        heatmap.py
        heuristic_detector.py
        pipeline.py
        video_processor.py
    simulation/
        crowd_simulator.py
    visualization/
        bird_eye_view.py
        crowd_3d.py
    export/
        video_generator.py
    assets/
    outputs/
```

## Troubleshooting

If video export fails with a codec error, upgrade OpenCV:

```
pip install opencv-python-headless --upgrade
```

If `ultralytics` fails to install or its weights fail to download, this is
expected to be handled gracefully. The app will run in heuristic or
simulated mode instead, and the interface will say so.

If disk space is limited, you can install only the core dependencies and
skip `ultralytics` entirely:

```
pip install streamlit opencv-python-headless numpy pandas plotly
```
