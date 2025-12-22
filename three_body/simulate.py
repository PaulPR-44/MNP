from __future__ import annotations
import numpy as np
from typing import Tuple, Callable

from .physics import pack_state, deriv, unpack_state, handle_collisions
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
    Returns: times (T,), R (T,N,3), V (T,N,3), state (T,2,N,3), M_hist (T,N), S_hist (T,N)
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    masses = np.asarray(masses, dtype=float).copy()
    sizes = np.asarray(sizes, dtype=float).copy()

    if not (r0.shape == v0.shape and r0.ndim == 2 and r0.shape[1] == 3):
        raise ValueError("r0 and v0 must be (N,3)")
    if masses.shape[0] != r0.shape[0]:
        raise ValueError("masses must be length N")

    n_bodies = len(masses)
    n_steps_guess = int(np.ceil(t_total / dt)) + 1
    M_hist = np.zeros((n_steps_guess, n_bodies))
    S_hist = np.zeros((n_steps_guess, n_bodies))

    def collision_handler(state: Array, t: float) -> tuple[Array, bool]:
        r, v = unpack_state(state)
        occurred = handle_collisions(masses, sizes, r, v)
        return pack_state(r, v), occurred

    def step_callback(state: Array, t: float, step_idx: int):
        if step_idx < len(M_hist):
            M_hist[step_idx] = masses
            S_hist[step_idx] = sizes

    state0 = pack_state(r0, v0)
    f = make_rhs(masses, softening=softening)
    times, Y = integrate(state0, t0, t0 + t_total, dt, f,
                         collision_handler=collision_handler,
                         step_callback=step_callback)

    # In case integrate returned fewer/more steps than guessed
    if len(times) != len(M_hist):
        M_hist = M_hist[:len(times)]
        S_hist = S_hist[:len(times)]

    R = Y[:, 0]
    V = Y[:, 1]
    return times, R, V, Y, M_hist, S_hist