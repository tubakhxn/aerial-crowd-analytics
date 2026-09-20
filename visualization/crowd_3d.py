# dev/creator: tubakhxn

import numpy as np
import plotly.graph_objects as go

from config import THEME

def _density_grid(points_xy, frame_w, frame_h, grid=(24, 14)):
    gx, gy = grid
    counts = np.zeros((gy, gx))
    if points_xy:
        cw, ch = frame_w / gx, frame_h / gy
        for x, y in points_xy:
            cx = min(int(x // cw), gx - 1)
            cy = min(int(y // ch), gy - 1)
            counts[cy, cx] += 1
    return counts

def build_3d_density_figure(points_xy, frame_w, frame_h, grid=(24, 14), mode="surface"):
    counts = _density_grid(points_xy, frame_w, frame_h, grid)
    gy, gx = counts.shape
    xs = np.linspace(0, frame_w, gx)
    ys = np.linspace(0, frame_h, gy)

    fig = go.Figure()

    if mode == "surface":
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=counts,
            colorscale=[[0, THEME["panel_bg"]], [0.35, THEME["blue"]],
                        [0.65, THEME["cyan"]], [1, THEME["red"]]],
            showscale=False, opacity=0.92,
        ))
    else:

        cw, ch = frame_w / gx, frame_h / gy
        xs_p, ys_p, zs_p, colors = [], [], [], []
        for x, y in points_xy:
            cx = min(int(x // cw), gx - 1)
            cy = min(int(y // ch), gy - 1)
            z = counts[cy, cx]
            xs_p.append(x)
            ys_p.append(y)
            zs_p.append(z)
            colors.append(z)
        fig.add_trace(go.Scatter3d(
            x=xs_p, y=ys_p, z=zs_p, mode="markers",
            marker=dict(size=3, color=colors,
                        colorscale=[[0, THEME["cyan"]], [0.6, THEME["orange"]], [1, THEME["red"]]],
                        opacity=0.85, showscale=False),
        ))

    fig.update_layout(
        paper_bgcolor=THEME["panel_bg"],
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(title="density", color=THEME["muted"], backgroundcolor=THEME["panel_bg"],
                        gridcolor=THEME["border"]),
            bgcolor=THEME["panel_bg"],
            camera=dict(eye=dict(x=1.4, y=-1.4, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
    )
    return fig
