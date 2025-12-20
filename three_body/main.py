from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, List, Sequence
import numpy as np

from .simulate import simulate
from .visualize import animate


def _parse_vec_list(arg: str) -> np.ndarray:
    """
    Parse a string like "x1:y1:z1;x2:y2:z2;..." or JSON-like list into (N,3) array.
    Also accepts comma-separated triples: "x1,y1,z1; x2,y2,z2".
    """
    s = arg.strip()
    # Try JSON first
    try:
        arr = np.asarray(json.loads(s), dtype=float)
        arr = arr.reshape((-1, 3))
        return arr
    except Exception:
        pass
    parts = [p.strip() for p in s.replace(" ", "").split(";") if p.strip()]
    triples = []
    for p in parts:
        if ":" in p:
            xyz = p.split(":")
        else:
            xyz = p.split(",")
        if len(xyz) != 3:
            raise argparse.ArgumentTypeError(f"Invalid vector triple: '{p}'")
        triples.append([float(x) for x in xyz])
    if not triples:
        raise argparse.ArgumentTypeError("No vectors provided")
    return np.array(triples, dtype=float)


def _parse_list(arg: str) -> np.ndarray:
    """Parse a JSON array or comma-separated list into 1D numpy array."""
    s = arg.strip()
    try:
        arr = np.asarray(json.loads(s), dtype=float)
        return arr.ravel()
    except Exception:
        pass
    items = [x for x in s.replace(" ", "").split(",") if x]
    if not items:
        raise argparse.ArgumentTypeError("Empty list")
    return np.array([float(x) for x in items], dtype=float)


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Three-body (N-body) simulator and visualizer")
    p.add_argument("--config", type=Path, help="Path to JSON config with masses, r0, v0, t_total[, dt]", default=None)

    # Direct parameters (overrides config if provided)
    p.add_argument("--masses", type=_parse_list, help="Comma/JSON list of masses (kg)")
    p.add_argument("--r0", type=_parse_vec_list, help="Initial positions (m): JSON [[x,y,z],...] or 'x,y,z;...'")
    p.add_argument("--v0", type=_parse_vec_list, help="Initial velocities (m/s): JSON [[vx,vy,vz],...] or 'vx,vy,vz;...'")
    p.add_argument("--t-total", type=float, help="Total simulated time (s)")
    p.add_argument("--dt", type=float, help="Time step (s). If omitted, computed as t_total/steps")
    p.add_argument("--steps", type=int, default=2000, help="Number of integration steps when --dt is omitted (default 2000)")
    p.add_argument("--softening", type=float, default=0.0, help="Softening length (m) to avoid singularities")

    # Visualization
    p.add_argument("--labels", type=str, default=None, help="Comma/JSON list of labels for bodies")
    p.add_argument("--dims", type=str, default="0,1", help="Which axes to plot, e.g. '0,1' for x-y or '0,2' for x-z")
    p.add_argument("--interval", type=int, default=20, help="Animation frame interval in ms")
    p.add_argument("--trail", type=int, default=0, help="Number of points in trailing path; 0 = full path (no erasing)")
    p.add_argument("--no-show", action="store_true", help="Do not display the animation window")
    p.add_argument("--save", type=Path, default=None, help="Path to save animation (mp4 or gif)")
    p.add_argument("--dpi", type=int, default=150, help="DPI for saved animation")
    p.add_argument("--writer", type=str, default=None, help="Animation writer: 'ffmpeg' or 'pillow'")

    # Dynamic autoscale options
    p.add_argument("--no-autoscale", action="store_true", help="Disable dynamic axis scaling during animation")
    p.add_argument("--autoscale-mode", type=str, default="cluster", choices=["cluster", "all"], help="Autoscale mode: 'cluster' ignores outliers via quantile; 'all' includes all bodies")
    p.add_argument("--autoscale-window", type=int, default=50, help="Frames window for autoscale (<=0 means use full history so far)")
    p.add_argument("--autoscale-quantile", type=float, default=0.98, help="Quantile for cluster bounds (0.5..1.0)")
    p.add_argument("--autoscale-margin", type=float, default=0.05, help="Fractional margin around data when autoscaling")
    p.add_argument("--autoscale-smooth", type=float, default=0.2, help="EMA smoothing factor for autoscale (0..1)")

    return p


def resolve_inputs(args: argparse.Namespace) -> dict:
    data: dict[str, Any] = {}
    if args.config:
        data.update(load_config(args.config))

    # Override with CLI values if provided
    if args.masses is not None:
        data["masses"] = np.asarray(args.masses, dtype=float).tolist()
    if args.r0 is not None:
        data["r0"] = np.asarray(args.r0, dtype=float).tolist()
    if args.v0 is not None:
        data["v0"] = np.asarray(args.v0, dtype=float).tolist()
    if args.t_total is not None:
        data["t_total"] = float(args.t_total)
    if args.dt is not None:
        data["dt"] = float(args.dt)
    if args.softening is not None:
        data["softening"] = float(args.softening)

    # Basic validation (dt may be computed later)
    required = ["masses", "size", "r0", "v0", "t_total"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"Missing required parameters: {', '.join(missing)}. Provide --config or CLI args.")

    # Compute dt if omitted: dt = t_total / steps
    if "dt" not in data or data["dt"] is None:
        steps = int(getattr(args, "steps", 2000))
        if steps <= 0:
            raise SystemExit("--steps must be a positive integer when --dt is omitted")
        data["dt"] = float(data["t_total"]) / float(steps)
        data["steps"] = steps
    else:
        # Still report steps corresponding to dt for info
        data["steps"] = int(max(1, round(float(data["t_total"]) / float(data["dt"]))))

    # Labels
    if args.labels:
        try:
            labels = json.loads(args.labels)
            if isinstance(labels, list):
                data["labels"] = labels
        except Exception:
            data["labels"] = [s for s in args.labels.split(",")]

    # dims
    dims = tuple(int(x) for x in args.dims.replace(" ", "").split(","))
    if len(dims) != 2 or not all(d in (0, 1, 2) for d in dims):
        raise SystemExit("--dims must be two integers from {0,1,2}, e.g. '0,1'")
    data["dims"] = dims

    data["interval_ms"] = int(args.interval)
    data["trail"] = int(args.trail)
    data["save_path"] = str(args.save) if args.save else None
    data["dpi"] = int(args.dpi)
    data["writer"] = args.writer
    data["show"] = not args.no_show

    # Autoscale options
    data["autoscale"] = not bool(args.no_autoscale)
    data["autoscale_mode"] = str(args.autoscale_mode)
    data["autoscale_window"] = int(args.autoscale_window)
    data["autoscale_quantile"] = float(args.autoscale_quantile)
    data["autoscale_margin"] = float(args.autoscale_margin)
    data["autoscale_smooth"] = float(args.autoscale_smooth)

    return data


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = resolve_inputs(args)

    masses = np.asarray(cfg["masses"], dtype=float)
    sizes = np.asarray(cfg["size"], dtype=float)
    r0 = np.asarray(cfg["r0"], dtype=float)
    v0 = np.asarray(cfg["v0"], dtype=float)
    t_total = float(cfg["t_total"])
    dt = float(cfg["dt"])
    softening = float(cfg.get("softening", 0.0))

    labels = cfg.get("labels")
    dims = tuple(cfg.get("dims", (0, 1)))
    interval_ms = int(cfg.get("interval_ms", 20))
    trail = int(cfg.get("trail", 0))
    save_path = cfg.get("save_path")
    dpi = int(cfg.get("dpi", 150))
    writer = cfg.get("writer")
    show = bool(cfg.get("show", True))

    # Autoscale config
    autoscale = bool(cfg.get("autoscale", True))
    autoscale_mode = str(cfg.get("autoscale_mode", "cluster"))
    autoscale_window = cfg.get("autoscale_window", 50)
    autoscale_quantile = float(cfg.get("autoscale_quantile", 0.98))
    autoscale_margin = float(cfg.get("autoscale_margin", 0.05))
    autoscale_smooth = float(cfg.get("autoscale_smooth", 0.2))

    times, R, V, _ = simulate(masses, sizes, r0, v0, t_total, dt, softening=softening)

    animate(times, R, labels=labels, dims=dims, interval_ms=interval_ms, trail=trail,
            save_path=save_path, dpi=dpi, writer=writer, show=show,
            autoscale=autoscale,
            autoscale_mode=autoscale_mode,
            autoscale_window=autoscale_window,
            autoscale_quantile=autoscale_quantile,
            autoscale_margin=autoscale_margin,
            autoscale_smooth=autoscale_smooth)


if __name__ == "__main__":
    main()
