from __future__ import annotations
import numpy as np

# Gravitational constant in SI units (m^3 kg^-1 s^-2)
G = 6.67430e-11


def pack_state(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Pack positions and velocities into a single state vector.
    r: (N, 3), v: (N, 3) -> state: (2, N, 3)
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    if not (r.shape == v.shape and r.ndim == 2 and r.shape[1] == 3):
        raise ValueError("r and v must be (N,3)")
    return np.stack((r, v), axis=0)


def unpack_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_state: returns (r, v)."""
    if not (state.ndim == 3 and state.shape[0] == 2 and state.shape[2] == 3):
        raise ValueError("state must be (2,N,3)")
    return state[0], state[1]


def accelerations(r: np.ndarray, m: np.ndarray, softening: float = 0.0) -> np.ndarray:
    """
    Compute gravitational accelerations on N bodies due to each other.
    r: (N,3) positions
    m: (N,) masses
    Returns a: (N,3)

    softening: optional Plummer softening length (meters) to avoid singularities.
    """
    N = r.shape[0]
    a = np.zeros_like(r)
    for i in range(N):
        # vector from i to others
        dr = r - r[i]
        dist2 = np.sum(dr**2, axis=1) + softening**2
        # avoid self-force
        dist2[i] = np.inf
        inv_r3 = (dist2 ** -1.5)
        # Newton's law: a_i = G * sum_{j!=i} m_j * (r_j - r_i) / |r_j - r_i|^3
        a[i] = G * np.sum((m[:, None] * dr) * inv_r3[:, None], axis=0)
    return a


def deriv(state: np.ndarray, m: np.ndarray, softening: float = 0.0) -> np.ndarray:
    """
    Time derivative of state for N-body under gravity.
    state: (2,N,3) with [0]=r, [1]=v
    returns dstate/dt: (2,N,3) with [0]=v, [1]=a
    """
    r, v = unpack_state(state)
    a = accelerations(r, m, softening=softening)
    return np.stack((v, a), axis=0)
