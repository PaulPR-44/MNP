from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from typing import Optional, Sequence

Array = np.ndarray

DEFAULT_COLORS = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def _auto_limits(R: Array, margin: float = 0.05):
    # R: (T,N,3)
    mins = R.min(axis=(0, 1))
    maxs = R.max(axis=(0, 1))
    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    rng = max(maxs[0] - mins[0], maxs[1] - mins[1])
    if rng == 0:
        rng = 1.0
    pad = margin * rng
    return (cx - 0.5 * rng - pad, cx + 0.5 * rng + pad), (cy - 0.5 * rng - pad, cy + 0.5 * rng + pad)


def animate(times: Array, R: Array, labels: Optional[Sequence[str]] = None, colors: Optional[Sequence[str]] = None,
            trail: Optional[int] = None, interval_ms: int = 20, equal_aspect: bool = True, dims: tuple[int, int] = (0, 1),
            save_path: Optional[str] = None, dpi: int = 150, writer: Optional[str] = None,
            show: bool = True,
            autoscale: bool = True,
            autoscale_mode: str = "cluster",
            autoscale_window: Optional[int] = 50,
            autoscale_quantile: float = 0.98,
            autoscale_margin: float = 0.05,
            autoscale_smooth: float = 0.2):
    """
    Create an animation of trajectories.
    - times: (T,)
    - R: (T,N,3) positions
    - labels: optional list of N labels
    - colors: optional list of N color specs
    - trail: number of points in the trailing path; if None or <= 0, show the full path from t=0
    - interval_ms: delay between frames in milliseconds
    - dims: which two axes to display (default x-y)
    - save_path: if provided, save animation to this file (mp4 or gif)
    - writer: 'ffmpeg' for mp4 or 'pillow' for gif (auto-inferred from extension if None)
    - show: whether to display the animation window
    - autoscale: dynamically adapt axes limits during the animation
    - autoscale_mode: 'all' uses all bodies; 'cluster' ignores extreme outliers (quantile clipping)
    - autoscale_window: number of recent frames to consider for scaling (None or <=0 = full history so far)
    - autoscale_quantile: high quantile for cluster bounds (e.g., 0.98 keeps inner 98%)
    - autoscale_margin: extra margin fraction around data
    - autoscale_smooth: EMA smoothing factor for center/range (0..1); higher = faster updates
    """
    T, N, _ = R.shape
    if labels is None:
        labels = [f"Body {i+1}" for i in range(N)]
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(N)]

    # Select dimensions
    x = R[:, :, dims[0]]
    y = R[:, :, dims[1]]

    # Figure setup
    fig, ax = plt.subplots(figsize=(6, 6))
    (xlim0, ylim0) = _auto_limits(R)
    ax.set_xlim(*xlim0)
    ax.set_ylim(*ylim0)
    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(["x", "y", "z"][dims[0]] + " (m)")
    ax.set_ylabel(["x", "y", "z"][dims[1]] + " (m)")
    # Time label anchored inside axes (top-left) with semi-transparent background to avoid overlap and maintain blitting compatibility
    time_text = ax.text(0.02, 0.98, "t = 0.0 s", transform=ax.transAxes,
                        ha="left", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
                        zorder=4)

    # Artists
    scatters = []
    trails = []
    for i in range(N):
        sc = ax.scatter([], [], s=30, color=colors[i], label=labels[i], zorder=3)
        ln, = ax.plot([], [], color=colors[i], lw=1.5, alpha=0.8, zorder=2)
        scatters.append(sc)
        trails.append(ln)
    ax.legend(loc='upper right')

    # Initialize autoscale state
    if autoscale_window is not None and isinstance(autoscale_window, (int, np.integer)) and autoscale_window <= 0:
        autoscale_window = None
    # Initial limits center and range for smoothing
    cx0 = 0.5 * (xlim0[0] + xlim0[1])
    cy0 = 0.5 * (ylim0[0] + ylim0[1])
    rx0 = 0.5 * (xlim0[1] - xlim0[0])
    ry0 = 0.5 * (ylim0[1] - ylim0[0])
    cxs, cys = cx0, cy0
    rxs, rys = rx0, ry0

    def _compute_limits(f: int):
        nonlocal cxs, cys, rxs, rys
        if not autoscale:
            return None
        w0 = 0 if autoscale_window is None else max(0, f - int(autoscale_window) + 1)
        xs = x[w0:f+1, :].reshape(-1)
        ys = y[w0:f+1, :].reshape(-1)
        if xs.size == 0:
            return None
        if autoscale_mode == 'cluster':
            q = float(autoscale_quantile)
            q = min(max(q, 0.5), 1.0)
            lo_q = (1.0 - q) / 2.0
            hi_q = 1.0 - lo_q
            xmin, xmax = np.quantile(xs, [lo_q, hi_q])
            ymin, ymax = np.quantile(ys, [lo_q, hi_q])
        else:
            xmin, xmax = float(np.min(xs)), float(np.max(xs))
            ymin, ymax = float(np.min(ys)), float(np.max(ys))
        # Avoid zero range
        if xmax == xmin:
            xmax = xmin + 1.0
        if ymax == ymin:
            ymax = ymin + 1.0
        # Apply margin
        mx = autoscale_margin * (xmax - xmin)
        my = autoscale_margin * (ymax - ymin)
        xmin -= mx
        xmax += mx
        ymin -= my
        ymax += my
        # Smooth center and ranges
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        rx = 0.5 * (xmax - xmin)
        ry = 0.5 * (ymax - ymin)
        alpha = float(autoscale_smooth)
        alpha = min(max(alpha, 0.0), 1.0)
        cxs = (1 - alpha) * cxs + alpha * cx
        cys = (1 - alpha) * cys + alpha * cy
        rxs = (1 - alpha) * rxs + alpha * rx
        rys = (1 - alpha) * rys + alpha * ry
        if equal_aspect:
            rr = max(rxs, rys)
            return (cxs - rr, cxs + rr), (cys - rr, cys + rr)
        else:
            return (cxs - rxs, cxs + rxs), (cys - rys, cys + rys)

    def init():
        for i in range(N):
            scatters[i].set_offsets(np.c_[[], []])
            trails[i].set_data([], [])
        time_text.set_text("t = 0.0 s")
        return [time_text, *scatters, *trails]

    def update(frame):
        t = times[frame]
        # If trail is None or <= 0, show full history from t=0
        t0 = 0 if (trail is None or (isinstance(trail, (int, np.integer)) and trail <= 0)) else max(0, frame - int(trail))
        time_text.set_text(f"t = {t:,.2f} s")
        # Update data
        for i in range(N):
            xi = x[t0:frame + 1, i]
            yi = y[t0:frame + 1, i]
            trails[i].set_data(xi, yi)
            scatters[i].set_offsets([x[frame, i], y[frame, i]])
        # Update limits
        lims = _compute_limits(frame)
        if lims is not None:
            (xlim, ylim) = lims
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        return [time_text, *scatters, *trails]

    blit_flag = not autoscale  # changing axes limits typically requires full redraw
    ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=blit_flag, interval=interval_ms)

    # Save if requested
    if save_path:
        if writer is None:
            if save_path.lower().endswith('.mp4'):
                writer = 'ffmpeg'
            elif save_path.lower().endswith('.gif'):
                writer = 'pillow'
            else:
                writer = 'ffmpeg'
        if writer == 'ffmpeg':
            ani.save(save_path, writer=FFMpegWriter(fps=max(1, int(1000/interval_ms))), dpi=dpi)
        elif writer == 'pillow':
            ani.save(save_path, writer=PillowWriter(fps=max(1, int(1000/interval_ms))), dpi=dpi)
        else:
            ani.save(save_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani
