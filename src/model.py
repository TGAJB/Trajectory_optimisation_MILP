"""
model.py - build and solve the MILP (Richards 2002, Sec 2.2 base LP + Sec 2.3 avoidance).

TODO - base LP:
    [x] Variables indexed by vehicle p, time step i, axis n (even when V = 1):
        x[p, i, :] states, u_plus[p, i, n] >= 0, u_minus[p, i, n] >= 0
        (u = u_plus - u_minus; the split is needed later for plumes)
    [x] Dynamics: x[i+1] = Ad x[i] + Bd u[i]
    [x] Boundary conditions: x[0] = start, x[T] = end
    [x] Bounds: |x| <= x_max, u_plus/u_minus <= u_max
    [x] Objective: minimise sum(u_plus + u_minus)  (fuel / 1-norm)
#THIS TODO IS PG 26 IN ARTICLE
    
TODO - obstacles (Sec 2.3.1):
    [x] outside_box(model, point, lower, upper, M): reusable big-M helper
        - 2N binaries a_k
        - point_n >= upper_n - M a_n
        - point_n <= lower_n + M a_(n+N)
        - sum(a) <= 2N - 1
    [x] add_obstacles(...): call outside_box for every vehicle, obstacle (inflated
        by obstacle_margin), and time step i = 1..T-1

TODO - later (stop with NotImplementedError until done):
    [ ] add_vehicle_avoidance(...): outside_box on x_p - x_q with box = +/- r, pairs q > p
    [ ] add_plumes(...): outside_box plus an extra "not firing" option
        (u_plus <= M c for forward thrusters, u_minus <= M c for reverse), 2N+1 binaries

TODO - solve:
    [x] build_and_solve(cfg, Ad, Bd): base LP -> obstacles -> (vehicles) -> (plumes)
    [x] Apply Gurobi params (TimeLimit, MIPGap)
    [x] Handle infeasible / time-limit status
    [x] Return trajectories, thrusts, fuel used, solve time, MIP gap, binary count
    [x] Print the binary count on every run
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB



# Base LP: fuel-optimal trajectory, no avoidance
#
# Index letters (used in general):
#   p  vehicle            0 .. V-1
#   i  time step          0 .. T   (states),  0 .. T-1 (thrusts)
#   k  state component    0 .. 2*dims-1   e.g. 2-D: [x, y, vx, vy]
#   j  state component, used as the column of Ad
#   n  thrust axis        0 .. dims-1

def add_base_lp(model, cfg, Ad, Bd):
    """
    Add the base fuel-optimal LP to `model`.

    Returns the variables so later steps (obstacles, solve) can use them:
        x[p, i, k]        state of vehicle p at time step i
        u_plus[p, i, n]   positive part of the thrust  (u = u_plus - u_minus)
        u_minus[p, i, n]  negative part of the thrust
    """
    #retrieve the data from the cfg
    general = cfg["general"]
    vehicles = cfg["vehicles"]

    V = len(vehicles)
    T = general["T"]
    dims = general["dims"]
    n_states = 2 * dims
    x_max = general["x_max"]
    u_max = general["u_max"]

    #VARIABLES AND BOUNDS
    # States: each component k has its own limit, so add them one by one.
    # Note: Gurobi's default lower bound is 0, so lb must be set explicitly.
    x = {}
    for p in range(V):
        for i in range(T + 1):
            for k in range(n_states):
                x[p, i, k] = model.addVar(lb=-x_max[k], ub=x_max[k], name=f"x[{p},{i},{k}]")

    # Thrusts: split u = u_plus - u_minus with both >= 0, so |u| = u_plus + u_minus
    # we dont want the solver to fires both at once, as it wastes fuel
    u_plus = model.addVars(V, T, dims, lb=0.0, ub=u_max, name="u_plus")
    u_minus = model.addVars(V, T, dims, lb=0.0, ub=u_max, name="u_minus")

    #Boundary conditions eq 1 and 2 from paper
    for p, vehicle in enumerate(vehicles):
        for k in range(n_states):
            model.addConstr(x[p, 0, k] == vehicle["start"][k], name=f"start[{p},{k}]")
            model.addConstr(x[p, T, k] == vehicle["end"][k], name=f"end[{p},{k}]")

    #Dynamics eq 3: x[i+1] = Ad x[i] + Bd u[i] (does it row by row)
    for p in range(V):
        for i in range(T):
            for k in range(n_states):
                drift = gp.quicksum(float(Ad[k, j]) * x[p, i, j] for j in range(n_states))
                thrust = gp.quicksum(float(Bd[k, n]) * (u_plus[p, i, n] - u_minus[p, i, n])
                                     for n in range(dims))
                model.addConstr(x[p, i + 1, k] == drift + thrust, name=f"dyn[{p},{i},{k}]")

    #Objective eq 7 and 8 -> MINIMIZE TOTAL FUEL -> sum of |u|
    model.setObjective(u_plus.sum() + u_minus.sum(), GRB.MINIMIZE)

    return x, u_plus, u_minus


# Avoidance constraints (Sec 2.3)

def outside_box(model, point, lower, upper, M, name):
    """
    Force `point` to lie outside the box [lower, upper] (Richards eq. 12), using big-M.

    point  list of N Gurobi variables/expressions, one per axis
           (later, for vehicle avoidance, this will be x_p - x_q)
    lower  N numbers, the box corner with the smallest coordinates
    upper  N numbers, the box corner with the largest coordinates

    "Outside" means at least one of these 2N conditions holds:
        point_n >= upper_n        (beyond the box on the positive side of axis n)
        point_n <= lower_n        (beyond the box on the negative side of axis n)

    Each condition gets a binary a. a = 0 enforces the condition; a = 1 switches it
    off by moving the bound M away. At most 2N-1 may be switched off, so at least
    one condition is enforced.

    Returns the list of 2N binaries.
    """
    N = len(point)
    a = model.addVars(2 * N, vtype=GRB.BINARY, name=name)

    for n in range(N):
        model.addConstr(point[n] >= upper[n] - M * a[n], name=f"{name}_above[{n}]")
        model.addConstr(point[n] <= lower[n] + M * a[n + N], name=f"{name}_below[{n}]")

    model.addConstr(a.sum() <= 2 * N - 1, name=f"{name}_one_side")

    return a


def add_obstacles(model, cfg, x):
    """
    Keep every vehicle outside every obstacle at every time step (Richards Sec 2.3.1).

    Obstacles are grown by obstacle_margin on every side, to reduce corner cutting
    between time steps (the constraints only hold AT the time steps).
    Only steps 1..T-1 are constrained: step 0 and step T are fixed by the start and end
    conditions, and config.py already checks those lie outside every obstacle.
    """
    general = cfg["general"]
    V = len(cfg["vehicles"])
    T = general["T"]
    dims = general["dims"]
    M = general["big_M"]
    margin = general.get("obstacle_margin", 0.0)

    for l, obstacle in enumerate(cfg["obstacles"]):
        # Inflated box
        lower = [obstacle["lower"][n] - margin for n in range(dims)]
        upper = [obstacle["upper"][n] + margin for n in range(dims)]

        for p in range(V):
            for i in range(1, T):
                # Position of vehicle p at step i = the first `dims` state components
                position = [x[p, i, n] for n in range(dims)]
                outside_box(model, position, lower, upper, M, name=f"obs[{p},{l},{i}]")


def add_vehicle_avoidance(model, cfg, x):
    raise NotImplementedError("Vehicle avoidance is not built yet (multi_vehicle)")


def add_plumes(model, cfg, x, u_plus, u_minus):
    raise NotImplementedError("Plume avoidance is not built yet (plumes)")


#SOLVER

# Readable names for the Gurobi status codes we expect to see
STATUS_NAMES = {
    GRB.OPTIMAL: "optimal",
    GRB.TIME_LIMIT: "time limit reached",
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "infeasible or unbounded",
    GRB.UNBOUNDED: "unbounded",
}


def build_and_solve(cfg, Ad, Bd):
    """
    Build the MILP -> returns a solved dictionary
        X           states,  shape (V, T+1, 2*dims)
        U           thrusts, shape (V, T, dims)       (u_plus - u_minus)
        fuel        objective value, sum of |u| over all vehicles/steps/axes  [N]
        delta_v     total delta-v over all vehicles = fuel * dt / mass        [m/s]
        status      readable Gurobi status, e.g. "optimal"
        runtime     solve time [s]
        mip_gap     relative gap between best solution and best bound (0 for a pure LP)
        n_binaries  number of binary variables in the model
    """
    general = cfg["general"]
    solver = general.get("solver", {})

    model = gp.Model("trajectory")

    # Solver settings: leave Gurobi's default when the config value is null
    if solver.get("time_limit") is not None:
        model.Params.TimeLimit = solver["time_limit"]
    if solver.get("mip_gap") is not None:
        model.Params.MIPGap = solver["mip_gap"]

    #base LP first, then the avoidance constraints that are switched on
    x, u_plus, u_minus = add_base_lp(model, cfg, Ad, Bd)

    if len(cfg["obstacles"]) > 0:
        add_obstacles(model, cfg, x)
    if cfg["vehicle_avoidance"]:
        add_vehicle_avoidance(model, cfg, x)
    if cfg["plumes"]:
        add_plumes(model, cfg, x, u_plus, u_minus)

    # Gurobi only counts the variables after an update
    model.update()
    print(f"Binary variables: {model.NumBinVars}")

    model.optimize()

    status = STATUS_NAMES.get(model.Status, f"Gurobi status code {model.Status}")

    # No solution at all (infeasible, or time limit hit before any solution was found)
    if model.SolCount == 0:
        raise RuntimeError(f"No solution found: {status}")

    # A solution exists but may not be optimal
    if model.Status == GRB.TIME_LIMIT:
        print(f"Time limit reached: returning best solution found, "
              f"within {100 * model.MIPGap:.2f}% of optimal")

    # Copy the solution into NumPy arrays
    V = len(cfg["vehicles"])
    T = general["T"]
    dims = general["dims"]
    n_states = 2 * dims

    X = np.zeros((V, T + 1, n_states))
    for p in range(V):
        for i in range(T + 1):
            for k in range(n_states):
                X[p, i, k] = x[p, i, k].X

    U = np.zeros((V, T, dims))
    for p in range(V):
        for i in range(T):
            for n in range(dims):
                U[p, i, n] = u_plus[p, i, n].X - u_minus[p, i, n].X

    # The MIP gap only exists once there are binaries (a pure LP has none)
    if model.IsMIP:
        mip_gap = model.MIPGap
    else:
        mip_gap = 0.0

    fuel = model.ObjVal
    delta_v = fuel * general["dt"] / general["mass"]

    return {
        "X": X,
        "U": U,
        "fuel": fuel,
        "delta_v": delta_v,
        "status": status,
        "runtime": model.Runtime,
        "mip_gap": mip_gap,
        "n_binaries": model.NumBinVars,
    }

