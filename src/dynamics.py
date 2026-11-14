"""
dynamics.py - continuous-time models and zero-order-hold discretisation.

TODO:
    [ ] double_integrator(dims, mass): continuous A, B for free space
        (state = [positions, velocities], input = thrust)
    [ ] hill_cw(omega, mass): continuous A, B for Hill's / Clohessy-Wiltshire
        equations (x radial, y in-track, z out-of-plane), see Richards eq. (4)
        - for dims = 2, decide whether to use the in-plane (x, y) subset
    [ ] discretise_zoh(A, B, dt): exact ZOH via the matrix exponential of
        the augmented matrix [[A, B], [0, 0]] * dt -> (Ad, Bd)
    [ ] get_discrete_model(cfg): pick the model from config and return (Ad, Bd)
"""

import numpy as np
from scipy.linalg import expm
