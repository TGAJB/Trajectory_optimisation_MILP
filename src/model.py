"""
model.py - build and solve the MILP (Richards 2002, Sec 2.2 base LP + Sec 2.3 avoidance).

TODO - base LP:
    [ ] Variables indexed by vehicle p, time step i, axis n (even when V = 1):
        x[p, i, :] states, u_plus[p, i, n] >= 0, u_minus[p, i, n] >= 0
        (u = u_plus - u_minus; the split is needed later for plumes)
    [ ] Dynamics: x[i+1] = Ad x[i] + Bd u[i]
    [ ] Boundary conditions: x[0] = start, x[T] = end
    [ ] Bounds: |x| <= x_max, u_plus/u_minus <= u_max
    [ ] Objective: minimise sum(u_plus + u_minus)  (fuel / 1-norm)

TODO - obstacles (Sec 2.3.1):
    [ ] outside_box(model, point, lower, upper, M): reusable big-M helper
        - 2N binaries a_k
        - point_n >= upper_n - M a_n
        - point_n <= lower_n + M a_(n+N)
        - sum(a) <= 2N - 1
    [ ] add_obstacles(...): call outside_box for every vehicle, obstacle (inflated
        by obstacle_margin), and time step i = 1..T-1

TODO - later (stop with NotImplementedError until done):
    [ ] add_vehicle_avoidance(...): outside_box on x_p - x_q with box = +/- r, pairs q > p
    [ ] add_plumes(...): outside_box plus an extra "not firing" option
        (u_plus <= M c for forward thrusters, u_minus <= M c for reverse), 2N+1 binaries

TODO - solve:
    [ ] build_and_solve(cfg, Ad, Bd): base LP -> obstacles -> (vehicles) -> (plumes)
    [ ] Apply Gurobi params (TimeLimit, MIPGap)
    [ ] Handle infeasible / time-limit status
    [ ] Return trajectories, thrusts, fuel used, solve time, MIP gap, binary count
    [ ] Print the binary count on every run
"""

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB
