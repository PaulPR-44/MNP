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
        dr = r - r[i]
        dist2 = np.sum(dr**2, axis=1) + softening**2
        dist2[i] = np.inf
        mask = (m > 0) & (~np.any(np.isnan(r), axis=1))
        inv_r3 = np.zeros(N)
        safe_mask = mask.copy()
        safe_mask[i] = False
        safe_mask &= (dist2 > 0)
        
        inv_r3[safe_mask] = (dist2[safe_mask] ** -1.5)
        
        terms = (m[:, None] * dr) * inv_r3[:, None]
        a[i] = G * np.sum(terms[safe_mask], axis=0)
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


def handle_collisions(m: np.ndarray, s: np.ndarray, r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Check for and resolve collisions.
    Returns (updated_m, updated_s, updated_r, updated_v, collision_occurred)
    Collided bodies are merged: one is updated, others are "deactivated" (mass=0, size=0).
    """
    N = r.shape[0]
    collision_occurred = False
    new_m = m.copy()
    new_s = s.copy()
    new_r = r.copy()
    new_v = v.copy()

    deactivated = np.where(new_m <= 0)[0].tolist()

    for i in range(N):
        if i in deactivated:
            continue
        for j in range(i + 1, N):
            if j in deactivated:
                continue
            
            dist = np.linalg.norm(new_r[i] - new_r[j])
            if dist <= (new_s[i] + new_s[j]):
                m1, m2 = new_m[i], new_m[j]
                v1, v2 = new_v[i], new_v[j]
                r1, r2 = new_r[i], new_r[j]
                
                total_m = m1 + m2
                new_v[i] = (m1 * v1 + m2 * v2) / total_m
                new_r[i] = (m1 * r1 + m2 * r2) / total_m
                new_m[i] = total_m
                new_s[i] = (new_s[i]**3 + new_s[j]**3)**(1/3)
                
                new_m[j] = 0.0
                new_s[j] = 0.0
                new_r[j] = np.nan
                new_v[j] = 0.0
                
                deactivated.append(j)
                collision_occurred = True
                
    return new_m, new_s, new_r, new_v, collision_occurred

def collide(m: np.ndarray, r:np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handle collisions in the state Y.
    Returns new positions and velocities after collisions.
    """
    uniques, uniques_idx, unique_counts = np.unique(r, axis=0, return_inverse=True, return_counts=True)
    collision_mask = uniques[unique_counts > 1]
    if collision_mask.size == 0:
        return m, r, v

    collision_idx = np.where(np.isin(uniques_idx, np.where(unique_counts > 1)[0]))[0]
    pos_mask = np.isin(np.arange(r.shape[0]), collision_idx)

    dup_indices = np.nonzero(pos_mask)[0]

    if len(dup_indices) < 2:
        return  m, r, v
    
    new_velocity = inelastic_collision(
        m[dup_indices[0]],
        m[dup_indices[1]],
        v[dup_indices[0]],
        v[dup_indices[1]]
    )
    v[dup_indices[0]] = new_velocity
    r[dup_indices[0]] = r[dup_indices[1]]  
    r = np.delete(r, dup_indices[1], axis=0)
    v = np.delete(v, dup_indices[1], axis=0)

    m[dup_indices[0]] = m[dup_indices[0]] + m[dup_indices[1]]  
    m = np.delete(m, dup_indices[1])  

    return  m, r, v

def inelastic_collision(m1: float, m2: float, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the post-collision velocity for two bodies with vector velocities.
    """
    return ((m1 * v1) + (m2 * v2)) / (m1 + m2)
