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


def deriv(state: np.ndarray, m: np.ndarray, s: np.ndarray, softening: float = 0.0) -> np.ndarray:
    """
    Time derivative of state for N-body under gravity.
    state: (2,N,3) with [0]=r, [1]=v
    returns dstate/dt: (2,N,3) with [0]=v, [1]=a
    """
    r, v = unpack_state(state)
    m_col, r_col, v_col = collide_with_size(m, s, r, v)
    a = accelerations(r_col, m_col, softening=softening)
    return np.stack((v_col, a), axis=0)


def collide_with_size(m: np.ndarray, s: np.ndarray, r:np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    to_remove = set()
    new_r = np.zeros_like(r)
    new_v = np.zeros_like(v)
    new_m = m
    N = r.shape[0]
    for i in range(N):
        new_r[i] = r[i]
        new_v[i] = v[i]
        for j in range(i+1, N):
            dist = np.linalg.norm(r[i] - r[j])
            if dist <=  (s[i] + s[j]):
                col = inelastic_collision(m[i],m[j],v[i],v[j])
                v[i] = col
                to_remove.add(j)
                m[i] = m[i] + m[j]

    if not to_remove:
        return m, r, v

    new_r = np.delete(r, list(to_remove), axis=0)
    new_v = np.delete(v, list(to_remove), axis=0)

    return new_m, new_r, new_v

def inelastic_collision(m1: float, m2: float, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the post-collision velocity for two bodies with vector velocities.
    """
    return ((m1 * v1) + (m2 * v2)) / (m1 + m2)
