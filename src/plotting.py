"""
plotting.py - visualise trajectories and obstacles.

TODO:
    [x] plot_2d(result, cfg): trajectory per vehicle, obstacles as rectangles
        (actual and inflated), start/end markers
    [x] plot_3d(result, cfg): same in 3-D with boxes drawn as Poly3DCollection
    [x] Thrust arrows at time steps where the vehicle fires
    [ ] Later: draw the safety box around vehicles and the plume regions
    [x] Optionally save the figure to results/   (pass save_path=...)

Usage:
    plot_2d(result, cfg)                                   # show on screen
    plot_2d(result, cfg, save_path="results/run.png")      # also save
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# One fixed colour per vehicle, in this order (vehicle 1 is always blue, etc.)
VEHICLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Obstacles are neutral grey so they never look like a vehicle
OBSTACLE_FILL = "#c9c8c1"
OBSTACLE_EDGE = "#6b6a63"
GRID_COLOR = "#e4e3dd"

# Arrows are drawn when the thrust is above this fraction of u_max
FIRING_THRESHOLD = 1e-3

# The longest arrow (thrust = u_max) is this fraction of the plot size
ARROW_FRACTION = 0.08


# =============================================================================
# Helpers
# =============================================================================

def _axis_labels(cfg):
    """Axis names; Hill's equations have a physical meaning for each axis."""
    if cfg["general"]["dynamics"] == "hill":
        return ["x (radial) [m]", "y (in-track) [m]", "z (out-of-plane) [m]"]
    return ["x [m]", "y [m]", "z [m]"]


def _arrow_length_per_newton(result, cfg):
    """Scale factor so a full-thrust arrow is ARROW_FRACTION of the plot size."""
    dims = cfg["general"]["dims"]
    positions = result["X"][:, :, :dims]

    # Plot size: largest extent of the trajectories and obstacles on any axis
    lows = positions.reshape(-1, dims).min(axis=0)
    highs = positions.reshape(-1, dims).max(axis=0)
    for obstacle in cfg["obstacles"]:
        lows = np.minimum(lows, obstacle["lower"])
        highs = np.maximum(highs, obstacle["upper"])
    plot_size = max(highs - lows)

    if plot_size == 0:      # start == end and no obstacles: nothing to scale against
        plot_size = 1.0

    return ARROW_FRACTION * plot_size / cfg["general"]["u_max"]


def _firing_steps(U_vehicle, u_max):
    """Time steps where this vehicle's thrust is noticeably non-zero."""
    thrust_size = np.linalg.norm(U_vehicle, axis=1)
    return np.where(thrust_size > FIRING_THRESHOLD * u_max)[0]


def _box_faces(lower, upper):
    """The 6 faces of a 3-D box, each as a list of 4 corner points."""
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],   # bottom
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],   # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],   # front
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],   # back
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],   # left
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],   # right
    ]


def _title(result, cfg):
    return f"{cfg['scenario']}   |   total delta-v = {result['delta_v']:.3g} m/s"


def _finish(fig, save_path, show):
    """Save and/or show the figure."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


# =============================================================================
# 2-D
# =============================================================================

def plot_2d(result, cfg, save_path=None, show=True):
    """Trajectories, obstacles (actual and inflated), start/end markers and thrust arrows."""
    X = result["X"]
    U = result["U"]
    u_max = cfg["general"]["u_max"]
    margin = cfg["general"].get("obstacle_margin", 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ---- Obstacles ----------------------------------------------------------
    for j, obstacle in enumerate(cfg["obstacles"]):
        lower = obstacle["lower"]
        upper = obstacle["upper"]
        width = upper[0] - lower[0]
        height = upper[1] - lower[1]

        # Only label the first one, so the legend has one "Obstacle" entry
        ax.add_patch(Rectangle((lower[0], lower[1]), width, height,
                               facecolor=OBSTACLE_FILL, edgecolor=OBSTACLE_EDGE, linewidth=1,
                               label="Obstacle" if j == 0 else None))

        if margin > 0:
            ax.add_patch(Rectangle((lower[0] - margin, lower[1] - margin),
                                   width + 2 * margin, height + 2 * margin,
                                   fill=False, edgecolor=OBSTACLE_EDGE, linewidth=1, linestyle="--",
                                   label="Obstacle + margin" if j == 0 else None))

    # ---- Vehicles -----------------------------------------------------------
    length_per_newton = _arrow_length_per_newton(result, cfg)

    for p, vehicle in enumerate(cfg["vehicles"]):
        color = VEHICLE_COLORS[p]
        name = vehicle.get("name", f"vehicle {p + 1}")
        pos_x = X[p, :, 0]
        pos_y = X[p, :, 1]

        # Trajectory: one dot per time step
        ax.plot(pos_x, pos_y, "-o", color=color, linewidth=2, markersize=4, label=name)

        # Start (square) and end (diamond), with a white ring so they stand out
        ax.plot(pos_x[0], pos_y[0], "s", color=color, markersize=10,
                markeredgecolor="white", markeredgewidth=1.5,
                label="Start" if p == 0 else None)
        ax.plot(pos_x[-1], pos_y[-1], "D", color=color, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5,
                label="End" if p == 0 else None)

        # Thrust arrows: drawn at the step where the thrust starts acting,
        # pointing in the thrust direction, length proportional to |u|
        fired = _firing_steps(U[p], u_max)
        if len(fired) > 0:
            ax.quiver(pos_x[fired], pos_y[fired],
                      U[p, fired, 0] * length_per_newton, U[p, fired, 1] * length_per_newton,
                      angles="xy", scale_units="xy", scale=1,
                      color=color, width=0.004,
                      label="Thrust" if p == 0 else None)

    # ---- Axes ---------------------------------------------------------------
    labels = _axis_labels(cfg)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(_title(result, cfg))
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    _finish(fig, save_path, show)
    return fig, ax


# =============================================================================
# 3-D
# =============================================================================

def plot_3d(result, cfg, save_path=None, show=True):
    """Same as plot_2d, in 3-D. Obstacles are drawn as see-through boxes."""
    X = result["X"]
    U = result["U"]
    u_max = cfg["general"]["u_max"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    # ---- Obstacles (inflation margin not drawn in 3-D, it clutters the view) --
    for j, obstacle in enumerate(cfg["obstacles"]):
        box = Poly3DCollection(_box_faces(obstacle["lower"], obstacle["upper"]),
                               facecolor=OBSTACLE_FILL, edgecolor=OBSTACLE_EDGE,
                               linewidth=0.8, alpha=0.35,
                               label="Obstacle" if j == 0 else None)
        ax.add_collection3d(box)

    # ---- Vehicles -----------------------------------------------------------
    length_per_newton = _arrow_length_per_newton(result, cfg)

    for p, vehicle in enumerate(cfg["vehicles"]):
        color = VEHICLE_COLORS[p]
        name = vehicle.get("name", f"vehicle {p + 1}")
        pos = X[p, :, :3]

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-o", color=color,
                linewidth=2, markersize=3, label=name)
        ax.scatter(*pos[0], marker="s", s=70, color=color, edgecolors="white",
                   label="Start" if p == 0 else None)
        ax.scatter(*pos[-1], marker="D", s=60, color=color, edgecolors="white",
                   label="End" if p == 0 else None)

        fired = _firing_steps(U[p], u_max)
        if len(fired) > 0:
            arrows = U[p, fired, :] * length_per_newton
            ax.quiver(pos[fired, 0], pos[fired, 1], pos[fired, 2],
                      arrows[:, 0], arrows[:, 1], arrows[:, 2],
                      color=color, linewidth=1.2, arrow_length_ratio=0.3,
                      label="Thrust" if p == 0 else None)

    # ---- Axes ---------------------------------------------------------------
    labels = _axis_labels(cfg)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(_title(result, cfg))
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")

    _finish(fig, save_path, show)
    return fig, ax
