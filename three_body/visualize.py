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
            trail: Optional[int] = None, interval_ms: int = 20, equal_aspect: bool = True,
            dims: tuple[int, int] = (0, 1),
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
    labels, colors = _prepare_labels_and_colors(N, labels, colors)
    x, y = R[:, :, dims[0]], R[:, :, dims[1]]

    fig, ax, time_text, scatters, trails = _setup_figure(R, N, labels, colors, dims, equal_aspect)

    autoscale_state = _initialize_autoscale_state(R, autoscale_window, equal_aspect)

    limit_computer = _create_limit_computer(
        x, y, autoscale, autoscale_state, autoscale_window, autoscale_mode,
        autoscale_quantile, autoscale_margin, autoscale_smooth, equal_aspect
    )

    init_func = _create_init_function(N, scatters, trails, time_text)
    update_func = _create_update_function(
        times, x, y, N, trail, time_text, scatters, trails, limit_computer, ax
    )

    blit_flag = not autoscale
    ani = FuncAnimation(fig, update_func, frames=T, init_func=init_func, blit=blit_flag, interval=interval_ms)

    _save_animation_if_requested(ani, save_path, writer, interval_ms, dpi)
    _show_or_close_figure(fig, show)

    return ani


def _prepare_labels_and_colors(N: int, labels: Optional[Sequence[str]], colors: Optional[Sequence[str]]) -> tuple[
    list[str], list[str]]:
    """Prepare default labels and colors if not provided."""
    if labels is None:
        labels = [f"Body {i + 1}" for i in range(N)]
    if colors is None:
        colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(N)]
    return list(labels), list(colors)


def _setup_figure(R: Array, N: int, labels: list[str], colors: list[str],
                  dims: tuple[int, int], equal_aspect: bool):
    """Create and configure the matplotlib figure and artists."""
    fig, ax = plt.subplots(figsize=(6, 6))
    xlim0, ylim0 = _auto_limits(R)
    ax.set_xlim(*xlim0)
    ax.set_ylim(*ylim0)

    if equal_aspect:
        ax.set_aspect('equal', adjustable='box')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel(["x", "y", "z"][dims[0]] + " (m)")
    ax.set_ylabel(["x", "y", "z"][dims[1]] + " (m)")

    time_text = ax.text(0.02, 0.98, "t = 0.0 s", transform=ax.transAxes,
                        ha="left", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
                        zorder=4)

    scatters, trails = _create_artists(ax, N, labels, colors)
    ax.legend(loc='upper right')

    return fig, ax, time_text, scatters, trails


def _create_artists(ax, N: int, labels: list[str], colors: list[str]):
    """Create scatter and trail plot artists for each body."""
    scatters = []
    trails = []
    for i in range(N):
        sc = ax.scatter([], [], s=30, color=colors[i], label=labels[i], zorder=3)
        ln, = ax.plot([], [], color=colors[i], lw=1.5, alpha=0.8, zorder=2)
        scatters.append(sc)
        trails.append(ln)
    return scatters, trails


def _initialize_autoscale_state(R: Array, autoscale_window: Optional[int], equal_aspect: bool) -> dict:
    """Initialize the autoscale state dictionary."""
    if autoscale_window is not None and isinstance(autoscale_window, (int, np.integer)) and autoscale_window <= 0:
        autoscale_window = None

    xlim0, ylim0 = _auto_limits(R)
    cx0 = 0.5 * (xlim0[0] + xlim0[1])
    cy0 = 0.5 * (ylim0[0] + ylim0[1])
    rx0 = 0.5 * (xlim0[1] - xlim0[0])
    ry0 = 0.5 * (ylim0[1] - ylim0[0])

    return {
        'cxs': cx0, 'cys': cy0,
        'rxs': rx0, 'rys': ry0
    }


def _create_limit_computer(x: Array, y: Array, autoscale: bool, autoscale_state: dict,
                           autoscale_window: Optional[int], autoscale_mode: str,
                           autoscale_quantile: float, autoscale_margin: float,
                           autoscale_smooth: float, equal_aspect: bool):
    """Create a closure that computes dynamic axis limits."""

    def compute_limits(f: int):
        if not autoscale:
            return None

        w0 = 0 if autoscale_window is None else max(0, f - int(autoscale_window) + 1)
        xs = x[w0:f + 1, :].reshape(-1)
        ys = y[w0:f + 1, :].reshape(-1)

        if xs.size == 0:
            return None

        xmin, xmax, ymin, ymax = _compute_data_bounds(xs, ys, autoscale_mode, autoscale_quantile)
        xmin, xmax, ymin, ymax = _apply_margin(xmin, xmax, ymin, ymax, autoscale_margin)

        _smooth_limits(autoscale_state, xmin, xmax, ymin, ymax, autoscale_smooth)

        return _build_limits(autoscale_state, equal_aspect)

    return compute_limits


def _compute_data_bounds(xs: Array, ys: Array, mode: str, quantile: float) -> tuple[float, float, float, float]:
    """Compute the bounds of the data using all points or quantile clipping."""
    if mode == 'cluster':
        q = min(max(float(quantile), 0.5), 1.0)
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

    return xmin, xmax, ymin, ymax


def _apply_margin(xmin: float, xmax: float, ymin: float, ymax: float, margin: float) -> tuple[
    float, float, float, float]:
    """Add margin around data bounds."""
    mx = margin * (xmax - xmin)
    my = margin * (ymax - ymin)
    return xmin - mx, xmax + mx, ymin - my, ymax + my


def _smooth_limits(state: dict, xmin: float, xmax: float, ymin: float, ymax: float, smooth: float):
    """Apply exponential moving average smoothing to center and range."""
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    rx = 0.5 * (xmax - xmin)
    ry = 0.5 * (ymax - ymin)

    alpha = min(max(float(smooth), 0.0), 1.0)
    state['cxs'] = (1 - alpha) * state['cxs'] + alpha * cx
    state['cys'] = (1 - alpha) * state['cys'] + alpha * cy
    state['rxs'] = (1 - alpha) * state['rxs'] + alpha * rx
    state['rys'] = (1 - alpha) * state['rys'] + alpha * ry


def _build_limits(state: dict, equal_aspect: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """Build final axis limits from smoothed state."""
    if equal_aspect:
        rr = max(state['rxs'], state['rys'])
        return (state['cxs'] - rr, state['cxs'] + rr), (state['cys'] - rr, state['cys'] + rr)
    else:
        return (state['cxs'] - state['rxs'], state['cxs'] + state['rxs']), \
            (state['cys'] - state['rys'], state['cys'] + state['rys'])


def _create_init_function(N: int, scatters: list, trails: list, time_text):
    """Create the animation initialization function."""

    def init():
        for i in range(N):
            scatters[i].set_offsets(np.c_[[], []])
            trails[i].set_data([], [])
        time_text.set_text("t = 0.0 s")
        return [time_text, *scatters, *trails]

    return init


def _create_update_function(times: Array, x: Array, y: Array, N: int, trail: Optional[int],
                            time_text, scatters: list, trails: list, limit_computer, ax):
    """Create the animation update function."""

    def update(frame):
        t = times[frame]
        t0 = 0 if (trail is None or (isinstance(trail, (int, np.integer)) and trail <= 0)) else max(0,
                                                                                                    frame - int(trail))

        time_text.set_text(f"t = {t:,.2f} s")

        for i in range(N):
            xi = x[t0:frame + 1, i]
            yi = y[t0:frame + 1, i]
            trails[i].set_data(xi, yi)
            scatters[i].set_offsets([x[frame, i], y[frame, i]])

        lims = limit_computer(frame)
        if lims is not None:
            xlim, ylim = lims
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        return [time_text, *scatters, *trails]

    return update


def _save_animation_if_requested(ani, save_path: Optional[str], writer: Optional[str],
                                 interval_ms: int, dpi: int):
    """Save the animation to file if a path is provided."""
    if not save_path:
        return

    if writer is None:
        writer = 'pillow' if save_path.lower().endswith('.gif') else 'ffmpeg'

    fps = max(1, int(1000 / interval_ms))

    if writer == 'ffmpeg':
        ani.save(save_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    elif writer == 'pillow':
        ani.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        ani.save(save_path, dpi=dpi)


def _show_or_close_figure(fig, show: bool):
    """Display or close the figure based on the show flag."""
    if show:
        plt.show()
    else:
        plt.close(fig)
