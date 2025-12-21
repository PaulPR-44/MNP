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
        if m[i] <= 0 or np.any(np.isnan(r[i])):
            continue

        # vector from i to others
        dr = r - r[i]
        dist2 = np.sum(dr**2, axis=1) + softening**2
        
        # avoid self-force and ignore deactivated bodies
        mask = (m > 0) & (~np.any(np.isnan(r), axis=1))
        mask[i] = False
        
        if not np.any(mask):
            continue

        inv_r3 = (dist2[mask] ** -1.5)
        # Newton's law: a_i = G * sum_{j!=i} m_j * (r_j - r_i) / |r_j - r_i|^3
        a[i] = G * np.sum((m[mask, None] * dr[mask]) * inv_r3[:, None], axis=0)
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


def handle_collisions(m: np.ndarray, s: np.ndarray, r: np.ndarray, v: np.ndarray) -> bool:
    """
    Detect and handle inelastic collisions between bodies.
    Modifies m, s, r, v in-place.
    Returns True if any collision occurred.
    """
    N = len(m)
    occurred = False
    for i in range(N):
        if m[i] <= 0 or np.any(np.isnan(r[i])):
            continue
        for j in range(i + 1, N):
            if m[j] <= 0 or np.any(np.isnan(r[j])):
                continue
            
            dr = r[i] - r[j]
            dist = np.linalg.norm(dr)
            if dist < (s[i] + s[j]):
                # Inelastic collision: conserve momentum
                v_new = (m[i] * v[i] + m[j] * v[j]) / (m[i] + m[j])
                # Combined radius (assuming constant density, volume adds up)
                s_new = (s[i]**3 + s[j]**3)**(1/3)
                
                # Update body i
                r[i] = (m[i] * r[i] + m[j] * r[j]) / (m[i] + m[j]) # Center of mass position
                v[i] = v_new
                m[i] = m[i] + m[j]
                s[i] = s_new
                
                # Deactivate body j
                m[j] = 0.0
                s[j] = 0.0
                r[j] = np.nan
                v[j] = np.nan
                
                occurred = True
    return occurred


def inelastic_collision(m1: float, m2: float, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the post-collision velocity for two bodies with vector velocities.
    """
    return ((m1 * v1) + (m2 * v2)) / (m1 + m2)
