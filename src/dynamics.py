"""
dynamics.py - continuous-time models and zero-order-hold discretisation.

TODO:
    [x] double_integrator(dims, mass): continuous A, B for free space
        (state = [positions, velocities], input = thrust)
    [x] hill_cw(omega, mass, dims): continuous A, B for Hill's / Clohessy-Wiltshire
        equations (x radial, y in-track, z out-of-plane), see Richards eq. (2.4)
        - for dims = 2, use the in-plane (x, y) subset
    [x] discretise_zoh(A, B, dt): exact ZOH via the matrix exponential of
        the augmented matrix [[A, B], [0, 0]] * dt -> (Ad, Bd)
    [x] get_discrete_model(cfg): pick the model from config and return (Ad, Bd)
        (get_continuous_model(cfg) gives the (A, B) it starts from)
"""

import numpy as np
from scipy.linalg import expm


def double_integrator(dims, mass):
    """
    Continuous-time free-space model  x_dot = A x + B u.

    State x = [positions..., velocities...]   (length 2 * dims)
    Input u = [thrust per axis...]            (length dims)
    This is Hill's equations (Richards eq. 2.4) with omega = 0.
    """
    n_states = 2 * dims
    n_inputs = dims
    identity = np.eye(dims)

    A = np.zeros((n_states, n_states))
    A[:dims, dims:] = identity               # gives the jordan ones in top right

    B = np.zeros((n_states, n_inputs))
    B[dims:, :] = identity / mass            # rows: velocities, columns: thrusts

    return A, B


def hill_cw(omega, mass, dims):
    #see pg25 of the article
    A, B = double_integrator(dims, mass)

    # Indices into the state vector (velocities start after the positions)
    x = 0
    vx, vy = dims, dims + 1 #row these terms have in the matrix

    # Radial acceleration: gravity gradient and Coriolis
    A[vx, x] = 3 * omega**2
    A[vx, vy] = 2 * omega

    # In-track acceleration: Coriolis
    A[vy, vx] = -2 * omega

    # Out-of-plane acceleration: restoring term (3-D only)
    if dims == 3:
        z = 2
        vz = dims + 2
        A[vz, z] = -omega**2

    return A, B

#zero order hold (u_i is held ct. over each step length dt)
def discretise_zoh(A, B, dt):
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n+m,n+m)) #M will house both A and B (shape is passed as one tuple)
    M[:n,:n] = A #A in top left block
    M[:n,n:] = B #B in top right block
    E = expm(M*dt) #exp(Mdt) basically
    Ad = E[:n,:n]
    Bd = E[:n,n:]
    return Ad,Bd #discretized blocks

def get_continuous_model(cfg):
    """Continuous (A, B) for the model chosen in the config (also used by the animation)."""
    general = cfg["general"]
    dims = general["dims"]
    mass = general["mass"]

    if general["dynamics"] == "double_integrator":
        A, B = double_integrator(dims, mass)
    elif general["dynamics"] == "hill":
        A, B = hill_cw(general["omega"], mass, dims)
    else:
        # config.validate() already checks this, but stop here too rather than
        # carrying on with no model
        raise ValueError(f"Unknown dynamics {general['dynamics']!r}")

    return A, B


def get_discrete_model(cfg):
    """Discrete (Ad, Bd) for one time step dt."""
    A, B = get_continuous_model(cfg)
    Ad, Bd = discretise_zoh(A, B, cfg["general"]["dt"])
    return Ad, Bd

