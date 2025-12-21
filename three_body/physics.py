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
    m_col, r_col, v_col = collide(m, r, v)
    a = accelerations(r_col, m_col, softening=softening)
    return np.stack((v_col, a), axis=0)

def collide(m: np.ndarray, r:np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handle collisions in the state Y.
    Returns new positions and velocities after collisions.
    """
    #print(f"Checking for collisions in {r.shape[0]} positions")
    uniques, uniques_idx, unique_counts = np.unique(r, axis=0, return_inverse=True, return_counts=True)
   # print(f"Found {len(uniques)} unique positions, with counts: {unique_counts}")
    collision_mask = uniques[unique_counts > 1]
    if collision_mask.size == 0:
        #print("No collisions detected.")
        return m, r, v

    collision_idx = np.where(np.isin(uniques_idx, np.where(unique_counts > 1)[0]))[0]
    pos_mask = np.isin(np.arange(r.shape[0]), collision_idx)

    dup_indices = np.nonzero(pos_mask)[0]
    print(f"Duplicate indices found: {dup_indices}")
    #print(f"Representing positions: {r[dup_indices]}")

    if len(dup_indices) < 2:
        print("Only one duplicate found, no collision to resolve.")
        return  m, r, v
    
    #print(f"Colliding bodies at indices {dup_indices} with new velocity {new_velocity}")

    new_velocity = inelastic_collision(
        m[dup_indices[0]],
        m[dup_indices[1]],
        v[dup_indices[0]],
        v[dup_indices[1]]
    )
   # print(f"Colliding bodies at indices {dup_indices} with new velocity {new_velocity:.3f}")
    # Update the velocity of the first duplicate
    v[dup_indices[0]] = new_velocity
    r[dup_indices[0]] = r[dup_indices[1]]  # Move the first duplicate to the position of the second
    # Remove the second duplicate
    r = np.delete(r, dup_indices[1], axis=0)
    v = np.delete(v, dup_indices[1], axis=0)

    m[dup_indices[0]] = m[dup_indices[0]] + m[dup_indices[1]]  # Combine the masses
    m = np.delete(m, dup_indices[1])  # Remove the mass of the second duplicate

    print(r)


    return  m, r, v

def inelastic_collision(m1: float, m2: float, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the post-collision velocity for two bodies with vector velocities.
    """
    return ((m1 * v1) + (m2 * v2)) / (m1 + m2)
