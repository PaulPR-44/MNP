from __future__ import annotations
import numpy as np
from typing import Tuple, Callable

from .physics import pack_state, unpack_state, deriv, handle_collisions
from .integrators import integrate

Array = np.ndarray


def make_rhs(masses: Array, softening: float = 0.0) -> Callable[[Array, float], Array]:
    """
    Create the RHS function f(state, t) for the N-body problem with given masses.
    state shape is (2,N,3).
    """
    def f(state: Array, t: float) -> Array:
        return deriv(state, masses, softening=softening)

    return f


def simulate(masses: Array, sizes: Array, r0: Array, v0: Array, t_total: float, dt: float, softening: float = 0.0,
             t0: float = 0.0) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Run a simulation and return times and trajectories.
    Inputs:
      - masses: (N,) kg 
      - sizes: (N,) m 
      - r0: (N,3) m
      - v0: (N,3) m/s
      - t_total: total simulated time (s)
      - dt: fixed time step (s)
      - softening: Plummer softening length (m)
    Returns: times (T,), R (T,N,3), V (T,N,3), state (T,2,N,3), M (T,N), S (T,N)
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    m = np.asarray(masses, dtype=float).copy()
    s = np.asarray(sizes, dtype=float).copy()
    
    if not (r0.shape == v0.shape and r0.ndim == 2 and r0.shape[1] == 3):
        raise ValueError("r0 and v0 must be (N,3)")
    if m.shape[0] != r0.shape[0]:
        raise ValueError("masses must be length N")

    state0 = pack_state(r0, v0)
    
    f = make_rhs(m, softening=softening)

    n_steps = int(np.ceil(t_total / dt)) + 1
    M_hist = np.zeros((n_steps, len(m)))
    S_hist = np.zeros((n_steps, len(s)))

    def step_callback(y: Array, t: float, i: int) -> None:
        if i < n_steps:
            M_hist[i] = m
            S_hist[i] = s

    def collision_handler(state: Array, t: float) -> tuple[Array, bool]:
        nonlocal m, s
        r, v = unpack_state(state)
        new_m, new_s, new_r, new_v, occurred = handle_collisions(m, s, r, v)
        if occurred:
            m[:] = new_m
            s[:] = new_s
            return pack_state(new_r, new_v), True
        return state, False

    times, Y = integrate(state0, t0, t0 + t_total, dt, f, collision_handler=collision_handler, step_callback=step_callback)
    
    if len(times) != len(M_hist):
        if len(M_hist) < len(times):
            M_hist = np.vstack([M_hist, m])
            S_hist = np.vstack([S_hist, s])
        else:
            M_hist = M_hist[:len(times)]
            S_hist = S_hist[:len(times)]

    R = Y[:, 0]
    V = Y[:, 1]
    return times, R, V, Y, M_hist, S_hist