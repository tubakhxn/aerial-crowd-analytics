# dev/creator: tubakhxn

import numpy as np
import plotly.graph_objects as go

from config import THEME

def build_bird_eye_figure(people_points, frame_w, frame_h, roi=None,
                           flow_angle=None, flow_mag=0.0, density_grid=None):
    fig = go.Figure()

    xs = [p[0] for p in people_points]
    ys = [frame_h - p[1] for p in people_points]

    if density_grid is not None and len(people_points) > 0:
        fig.add_trace(go.Histogram2dContour(
            x=[p[0] for p in people_points],
            y=[frame_h - p[1] for p in people_points],
            colorscale=[[0, THEME["panel_bg"]], [0.4, "#1b3a52"], [0.7, THEME["cyan"]], [1, THEME["green"]]],
            showscale=False,
            ncontours=12,
            opacity=0.55,
            line=dict(width=0),
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(size=5, color=THEME["green"], opacity=0.85,
                    line=dict(width=0.5, color=THEME["cyan"])),
        name="people", hoverinfo="skip",
    ))

    if roi is not None:
        x1, y1, x2, y2 = roi
        ry1, ry2 = frame_h - y1, frame_h - y2
        fig.add_shape(type="rect", x0=x1, y0=ry2, x1=x2, y1=ry1,
                      line=dict(color=THEME["cyan"], width=2), fillcolor="rgba(0,0,0,0)")

    if flow_angle is not None and flow_mag > 0.05:
        cx, cy = frame_w * 0.5, frame_h * 0.12
        rad = np.radians(flow_angle)
        dx, dy = np.cos(rad) * 60, -np.sin(rad) * 60
        fig.add_annotation(x=cx + dx, y=cy + dy, ax=cx, ay=cy,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.5,
                            arrowwidth=2, arrowcolor=THEME["orange"])

    fig.update_layout(
        paper_bgcolor=THEME["panel_bg"], plot_bgcolor=THEME["panel_bg"],
        margin=dict(l=4, r=4, t=4, b=4), showlegend=False,
        xaxis=dict(range=[0, frame_w], visible=False),
        yaxis=dict(range=[0, frame_h], visible=False, scaleanchor="x"),
        height=260,
    )
    return fig
