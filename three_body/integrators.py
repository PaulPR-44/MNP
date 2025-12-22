from __future__ import annotations
import numpy as np
from typing import Callable

Array = np.ndarray


def rk4_step(y: Array, t: float, dt: float, f: Callable[[Array, float], Array]) -> Array:
    """
    Classic fourth-order Runge-Kutta step for generic ODE y' = f(y, t).
    y: current state (any shape)
    t: current time
    dt: timestep
    f: function returning dy/dt with same shape as y
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(y0: Array, t0: float, tf: float, dt: float, f: Callable[[Array, float], Array],
              collision_handler: Callable[[Array, float], tuple[Array, bool]] | None = None,
              step_callback: Callable[[Array, float, int], None] | None = None) -> tuple[Array, Array]:
    """
    Integrate ODE using fixed-step RK4.
    Returns times array shape (T,) and states array shape (T, *y0.shape)
    Optional collision_handler(y, t) -> (new_y, occurred)
    Optional step_callback(y, t, step_idx)
    """
    if dt <= 0:
        raise ValueError("dt > 0")
    if tf <= t0:
        raise ValueError("tf must be > t0")
    n_steps = int(np.ceil((tf - t0) / dt))
    times = t0 + np.arange(n_steps + 1) * dt
    times[-1] = tf  # ensure exact final time

    y = y0.copy()
    Y = np.empty((len(times),) + y0.shape, dtype=float)
    Y[0] = y
    t = t0

    if step_callback is not None:
        step_callback(y, t, 0)

    for i in range(1, len(times)):
        this_dt = times[i] - t
        y = rk4_step(y, t, this_dt, f)
        t = times[i]
        
        if collision_handler is not None:
            y, _ = collision_handler(y, t)
            
        Y[i] = y
        if step_callback is not None:
            step_callback(y, t, i)

    return times, Y
