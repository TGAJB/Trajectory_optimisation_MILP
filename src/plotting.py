"""
plotting.py - visualise trajectories and obstacles.

TODO:
    [x] plot_2d(result, cfg): trajectory per vehicle, obstacles as rectangles
        (actual and inflated), start/end markers
    [x] plot_3d(result, cfg): same in 3-D with boxes drawn as Poly3DCollection
    [x] Thrust arrows at time steps where the vehicle fires
    [x] Optionally save the figure to results/   (pass save_path=...)
    [x] animate(result, cfg): vehicles moving along their paths, with
        - the safety box around each vehicle (multi_vehicle)
        - the plume box behind every thruster that fires
        - positions BETWEEN time steps from the real dynamics, so between-step
          corner cutting shows up honestly
        - a live readout: closest pair (in units of safety distance) and plume hits

Usage:
    plot_2d(result, cfg)                                   # show on screen
    plot_2d(result, cfg, save_path="results/run.png")      # also save
    anim = animate(result, cfg)                            # 2-D or 3-D, picked from dims
    anim = animate(result, cfg, save_path="results/run.gif")
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src import dynamics
from src.model import plume_box


# One fixed colour per vehicle, in this order (vehicle 1 is always blue, etc.)
VEHICLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Obstacles are neutral grey so they never look like a vehicle
OBSTACLE_FILL = "#c9c8c1"
OBSTACLE_EDGE = "#6b6a63"
GRID_COLOR = "#e4e3dd"

# Thrust arrows and plumes: one colour for all vehicles, slightly see-through
THRUST_COLOR = "#e34948"
THRUST_ALPHA = 0.6

# Arrows/plumes are drawn when the thrust is above this fraction of u_max
FIRING_THRESHOLD = 1e-3

# The longest arrow (thrust = u_max) is this fraction of the plot size
ARROW_FRACTION = 0.08

# ---- Animation ----
SUBSTEPS = 5                # frames per time step (positions in between come from the dynamics)
FRAME_INTERVAL_MS = 60      # time between frames on screen
PLUME_ALPHA_MIN = 0.12      # plume opacity at almost no thrust ...
PLUME_ALPHA_MAX = 0.45      # ... and at full thrust
WARNING_COLOR = "#1f1f1e"   # thick near-black outline: safety box breached / plume hits a vehicle


# =============================================================================
# Helpers
# =============================================================================

def _axis_labels(cfg):
    """Axis names; Hill's equations have a physical meaning for each axis."""
    if cfg["general"]["dynamics"] == "hill":
        return ["x (radial) [m]", "y (in-track) [m]", "z (out-of-plane) [m]"]
    return ["x [m]", "y [m]", "z [m]"]


def _vehicle_name(vehicle, p):
    return vehicle.get("name", f"vehicle {p + 1}")


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


def _equal_cube_axes(ax):
    """
    Give all three axes the same length, centred on the data, so distances look the
    same in every direction. A flat path (e.g. z = 0 everywhere) then still gets a
    readable z axis instead of one squashed to zero height.
    """
    limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    centres = [(low + high) / 2 for low, high in limits]
    half_size = max(high - low for low, high in limits) / 2

    ax.set_xlim3d(centres[0] - half_size, centres[0] + half_size)
    ax.set_ylim3d(centres[1] - half_size, centres[1] + half_size)
    ax.set_zlim3d(centres[2] - half_size, centres[2] + half_size)
    ax.set_box_aspect((1, 1, 1))


def _draw_obstacles_2d(ax, cfg):
    """Obstacles as grey rectangles, plus a dashed outline of the inflated box."""
    margin = cfg["general"].get("obstacle_margin", 0.0)

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


def _draw_obstacles_3d(ax, cfg):
    """Obstacles as very see-through boxes (inflation margin not drawn, it clutters the view)."""
    for j, obstacle in enumerate(cfg["obstacles"]):
        box = Poly3DCollection(_box_faces(obstacle["lower"], obstacle["upper"]),
                               facecolor=OBSTACLE_FILL, edgecolor=OBSTACLE_EDGE,
                               linewidth=0.6, alpha=0.12,      # walls must not hide the path
                               label="Obstacle" if j == 0 else None)
        ax.add_collection3d(box)


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

    fig, ax = plt.subplots(figsize=(8, 6))

    _draw_obstacles_2d(ax, cfg)

    # ---- Vehicles -----------------------------------------------------------
    length_per_newton = _arrow_length_per_newton(result, cfg)

    for p, vehicle in enumerate(cfg["vehicles"]):
        color = VEHICLE_COLORS[p]
        pos_x = X[p, :, 0]
        pos_y = X[p, :, 1]

        # Trajectory: one dot per time step
        ax.plot(pos_x, pos_y, "-o", color=color, linewidth=2, markersize=4,
                label=_vehicle_name(vehicle, p))

        # Start (square) and end (diamond), with a white ring so they stand out
        ax.plot(pos_x[0], pos_y[0], "s", color=color, markersize=10,
                markeredgecolor="white", markeredgewidth=1.5,
                label="Start" if p == 0 else None)
        ax.plot(pos_x[-1], pos_y[-1], "D", color=color, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5,
                label="End" if p == 0 else None)

        # Thrust arrows: the arrow tip sits on the dot of the step where the thrust
        # starts acting, pointing in the thrust direction, length proportional to |u|
        fired = _firing_steps(U[p], u_max)
        if len(fired) > 0:
            ax.quiver(pos_x[fired], pos_y[fired],
                      U[p, fired, 0] * length_per_newton, U[p, fired, 1] * length_per_newton,
                      angles="xy", scale_units="xy", scale=1, pivot="tip",
                      color=THRUST_COLOR, alpha=THRUST_ALPHA, width=0.004,
                      label="Thrust" if p == 0 else None)

            # Arrows don't widen the axes by themselves, so add their tails to the view
            tail_x = pos_x[fired] - U[p, fired, 0] * length_per_newton
            tail_y = pos_y[fired] - U[p, fired, 1] * length_per_newton
            ax.update_datalim(np.column_stack([tail_x, tail_y]))
            ax.autoscale_view()

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

    _draw_obstacles_3d(ax, cfg)

    # ---- Vehicles -----------------------------------------------------------
    length_per_newton = _arrow_length_per_newton(result, cfg)

    for p, vehicle in enumerate(cfg["vehicles"]):
        color = VEHICLE_COLORS[p]
        pos = X[p, :, :3]

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-o", color=color,
                linewidth=2, markersize=3, label=_vehicle_name(vehicle, p))
        ax.scatter(*pos[0], marker="s", s=70, color=color, edgecolors="white",
                   label="Start" if p == 0 else None)
        ax.scatter(*pos[-1], marker="D", s=60, color=color, edgecolors="white",
                   label="End" if p == 0 else None)

        fired = _firing_steps(U[p], u_max)
        if len(fired) > 0:
            arrows = U[p, fired, :] * length_per_newton
            ax.quiver(pos[fired, 0], pos[fired, 1], pos[fired, 2],
                      arrows[:, 0], arrows[:, 1], arrows[:, 2],
                      pivot="tip", color=THRUST_COLOR, alpha=THRUST_ALPHA,
                      linewidth=1.2, arrow_length_ratio=0.3,
                      label="Thrust" if p == 0 else None)

            # Arrows don't widen the axes by themselves, so add their tails to the view
            tails = pos[fired] - arrows
            ax.auto_scale_xyz(tails[:, 0], tails[:, 1], tails[:, 2], had_data=True)

    # ---- Axes ---------------------------------------------------------------
    labels = _axis_labels(cfg)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(_title(result, cfg))
    _equal_cube_axes(ax)
    ax.view_init(elev=20, azim=-20)      # side-on to the in-track (y) motion; drag to rotate
    ax.legend(frameon=False, loc="upper left")

    _finish(fig, save_path, show)
    return fig, ax


# =============================================================================
# Animation: what happens at every moment, including between time steps
# =============================================================================

def _fine_positions(result, cfg, substeps):
    """
    Positions at `substeps` points per time step, shape (V, T*substeps + 1, dims).

    The solver only knows the positions AT the time steps. In between, each vehicle
    is propagated with the real dynamics and the solved (constant) thrust of that
    step, so any corner cutting between steps is shown as it really happens.
    """
    general = cfg["general"]
    dims = general["dims"]
    X = result["X"]
    U = result["U"]
    V, T = U.shape[0], U.shape[1]

    A, B = dynamics.get_continuous_model(cfg)
    Ad_small, Bd_small = dynamics.discretise_zoh(A, B, general["dt"] / substeps)

    positions = np.zeros((V, T * substeps + 1, dims))
    for p in range(V):
        positions[p, 0] = X[p, 0, :dims]
        for i in range(T):
            state = X[p, i].copy()          # restart from the solved state at every step
            for s in range(1, substeps + 1):
                state = Ad_small @ state + Bd_small @ U[p, i]
                positions[p, i * substeps + s] = state[:dims]

    return positions


def _thrust_at_frame(U, frame, substeps):
    """Thrust of every vehicle during this frame, shape (V, dims). Zero in the final frame."""
    step = frame // substeps
    if step >= U.shape[1]:
        return np.zeros((U.shape[0], U.shape[2]))
    return U[:, step, :]


def _plumes_defined(cfg):
    plume = cfg.get("plume") or {}
    return plume.get("length") is not None and plume.get("half_width") is not None


def _active_plumes(position, thrust, cfg):
    """
    Plume boxes of the thrusters that fire right now, as a list of
    (lower, upper, strength), with strength = |thrust| / u_max between 0 and 1.
    At most one thruster per axis fires (the optimiser never fires both + and -).
    """
    dims = cfg["general"]["dims"]
    u_max = cfg["general"]["u_max"]
    length = cfg["plume"]["length"]
    half_width = cfg["plume"]["half_width"]

    plumes = []
    for n in range(dims):
        if abs(thrust[n]) <= FIRING_THRESHOLD * u_max:
            continue
        sign = 1 if thrust[n] > 0 else -1
        lower, upper = plume_box(n, sign, length, half_width, dims)
        plumes.append((position + np.array(lower), position + np.array(upper),
                       abs(thrust[n]) / u_max))
    return plumes


def _closest_pair(now, r):
    """
    Closest pair of vehicles, measured in safety distances: max_n |x_p - x_q|_n / r_n.
    1.0 means just touching the safety box, below 1.0 means it is breached.
    Returns (value, p, q).
    """
    best = (np.inf, None, None)
    for p in range(len(now)):
        for q in range(p + 1, len(now)):
            value = np.max(np.abs(now[p] - now[q]) / r)
            if value < best[0]:
                best = (value, p, q)
    return best


def _plume_hits(now, plumes):
    """List of (p, k, q): vehicle q sits strictly inside plume k of vehicle p."""
    hits = []
    for p, vehicle_plumes in enumerate(plumes):
        for k, (lower, upper, _strength) in enumerate(vehicle_plumes):
            for q in range(len(now)):
                if q != p and np.all(now[q] > lower) and np.all(now[q] < upper):
                    hits.append((p, k, q))
    return hits


def _frame_state(positions, U, frame, substeps, cfg, r):
    """Everything the animation needs to draw one frame."""
    now = positions[:, frame, :]
    thrust = _thrust_at_frame(U, frame, substeps)

    if _plumes_defined(cfg):
        plumes = [_active_plumes(now[p], thrust[p], cfg) for p in range(len(now))]
    else:
        plumes = [[] for _ in range(len(now))]

    closest = _closest_pair(now, r) if r is not None else None
    hits = _plume_hits(now, plumes)

    return now, plumes, closest, hits


def _readout(frame, substeps, cfg, names, closest, hits):
    """Text in the corner: time, closest pair, plume hits."""
    dt = cfg["general"]["dt"]
    T = cfg["general"]["T"]
    lines = [f"t = {frame * dt / substeps:5.0f} s   (step {frame / substeps:4.1f} / {T})"]

    if closest is not None:
        value, p, q = closest
        status = "OK" if value >= 1.0 - 1e-6 else "SAFETY BOX BREACHED"
        lines.append(f"closest pair {names[p]}-{names[q]}: {value:.2f} x safety distance  {status}")

    if _plumes_defined(cfg):
        constrained = "" if cfg["plumes"] else "   (plume constraints OFF)"
        if hits:
            hit_text = ", ".join(f"{names[p]} -> {names[q]}" for p, _k, q in hits)
            lines.append(f"plume hits: {hit_text}{constrained}")
        else:
            lines.append(f"plume hits: none{constrained}")

    return "\n".join(lines)


def _animation_limits(positions, cfg, r):
    """Lowest and highest corner of the view: paths, obstacles, plus room for boxes/plumes."""
    dims = cfg["general"]["dims"]
    lows = positions.reshape(-1, dims).min(axis=0)
    highs = positions.reshape(-1, dims).max(axis=0)
    for obstacle in cfg["obstacles"]:
        lows = np.minimum(lows, obstacle["lower"])
        highs = np.maximum(highs, obstacle["upper"])

    pad = 0.5
    if r is not None:
        pad = max(pad, max(r))
    if _plumes_defined(cfg):
        pad = max(pad, cfg["plume"]["length"])

    return lows - pad, highs + pad


def _legend_extras(r, cfg):
    """Legend entries for things that only exist while animating."""
    extras = []
    if r is not None:
        extras.append(Line2D([], [], color="#6b6a63", linestyle="--", label="Safety box"))
    if _plumes_defined(cfg):
        extras.append(Patch(facecolor=THRUST_COLOR, alpha=PLUME_ALPHA_MAX, label="Plume (firing)"))
    extras.append(Line2D([], [], color=WARNING_COLOR, linewidth=2.5, label="Breach / hit"))
    return extras


def _plume_alpha(strength):
    return PLUME_ALPHA_MIN + (PLUME_ALPHA_MAX - PLUME_ALPHA_MIN) * strength


def _finish_animation(anim, save_path, show):
    """Save as .gif (Pillow ships with matplotlib) and/or show."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer="pillow", fps=round(1000 / FRAME_INTERVAL_MS))
    if show:
        plt.show()


def animate_2d(result, cfg, substeps=SUBSTEPS, save_path=None, show=True):
    """Vehicles moving along their paths, with safety boxes and firing plumes."""
    U = result["U"]
    vehicles = cfg["vehicles"]
    V = len(vehicles)
    names = [_vehicle_name(vehicle, p) for p, vehicle in enumerate(vehicles)]
    r = np.array(cfg["safety_distance"]) if cfg["vehicle_avoidance"] else None

    positions = _fine_positions(result, cfg, substeps)
    n_frames = positions.shape[1]

    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_obstacles_2d(ax, cfg)

    # ---- Artists that stay, and artists that move every frame ---------------
    trails, dots, safety_boxes, plume_patches = [], [], [], []
    for p in range(V):
        color = VEHICLE_COLORS[p]

        # Whole planned path, faint, so you can see where each vehicle is heading
        ax.plot(positions[p, :, 0], positions[p, :, 1], color=color, linewidth=1, alpha=0.25)
        ax.plot(*positions[p, 0], "s", color=color, markersize=7, alpha=0.6)
        ax.plot(*positions[p, -1], "D", color=color, markersize=6, alpha=0.6)

        trail, = ax.plot([], [], color=color, linewidth=2)
        dot, = ax.plot([], [], "o", color=color, markersize=9,
                       markeredgecolor="white", markeredgewidth=1.5, zorder=5, label=names[p])
        trails.append(trail)
        dots.append(dot)

        if r is not None:
            box = Rectangle((0, 0), 2 * r[0], 2 * r[1], fill=False, edgecolor=color,
                            linewidth=1, linestyle="--", zorder=4)
            ax.add_patch(box)
            safety_boxes.append(box)

        # One plume patch per axis (only one thruster per axis fires at a time)
        patches = []
        for _n in range(2):
            patch = Rectangle((0, 0), 0, 0, facecolor=THRUST_COLOR, edgecolor=THRUST_COLOR,
                              linewidth=0.5, visible=False, zorder=3)
            ax.add_patch(patch)
            patches.append(patch)
        plume_patches.append(patches)

    readout = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                      fontsize=9, family="monospace", color="#2b2b29",
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    # ---- Axes ---------------------------------------------------------------
    lows, highs = _animation_limits(positions, cfg, r)
    ax.set_xlim(lows[0], highs[0])
    ax.set_ylim(lows[1], highs[1])
    ax.set_aspect("equal", adjustable="box")
    labels = _axis_labels(cfg)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(_title(result, cfg))
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + _legend_extras(r, cfg), frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()          # makes room for the legend outside the axes

    # ---- One frame ----------------------------------------------------------
    def update(frame):
        now, plumes, closest, hits = _frame_state(positions, U, frame, substeps, cfg, r)
        breached = set()
        if closest is not None and closest[0] < 1.0 - 1e-6:
            breached = {closest[1], closest[2]}
        hit_plumes = {(p, k) for p, k, _q in hits}

        for p in range(V):
            trails[p].set_data(positions[p, :frame + 1, 0], positions[p, :frame + 1, 1])
            dots[p].set_data([now[p, 0]], [now[p, 1]])

            if r is not None:
                box = safety_boxes[p]
                box.set_xy(now[p] - r)
                if p in breached:
                    box.set_edgecolor(WARNING_COLOR)
                    box.set_linewidth(2.5)
                else:
                    box.set_edgecolor(VEHICLE_COLORS[p])
                    box.set_linewidth(1)

            for k, patch in enumerate(plume_patches[p]):
                if k >= len(plumes[p]):
                    patch.set_visible(False)
                    continue
                lower, upper, strength = plumes[p][k]
                patch.set_xy(lower)
                patch.set_width(upper[0] - lower[0])
                patch.set_height(upper[1] - lower[1])
                patch.set_alpha(_plume_alpha(strength))
                if (p, k) in hit_plumes:
                    patch.set_edgecolor(WARNING_COLOR)
                    patch.set_linewidth(2.5)
                else:
                    patch.set_edgecolor(THRUST_COLOR)
                    patch.set_linewidth(0.5)
                patch.set_visible(True)

        readout.set_text(_readout(frame, substeps, cfg, names, closest, hits))

    anim = FuncAnimation(fig, update, frames=n_frames, interval=FRAME_INTERVAL_MS)
    _finish_animation(anim, save_path, show)
    return anim


def animate_3d(result, cfg, substeps=SUBSTEPS, save_path=None, show=True):
    """Same as animate_2d, in 3-D: safety cubes as dashed wireframes, plumes as boxes."""
    U = result["U"]
    vehicles = cfg["vehicles"]
    V = len(vehicles)
    names = [_vehicle_name(vehicle, p) for p, vehicle in enumerate(vehicles)]
    r = np.array(cfg["safety_distance"]) if cfg["vehicle_avoidance"] else None

    positions = _fine_positions(result, cfg, substeps)
    n_frames = positions.shape[1]

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(projection="3d")
    _draw_obstacles_3d(ax, cfg)

    # ---- Artists that stay, and artists that move every frame ---------------
    trails, dots, safety_boxes, plume_boxes = [], [], [], []
    for p in range(V):
        color = VEHICLE_COLORS[p]
        pos = positions[p]

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=1, alpha=0.25)
        ax.scatter(*pos[0], marker="s", s=40, color=color, alpha=0.6)
        ax.scatter(*pos[-1], marker="D", s=35, color=color, alpha=0.6)

        trail, = ax.plot([], [], [], color=color, linewidth=2)
        dot, = ax.plot([], [], [], "o", color=color, markersize=8,
                       markeredgecolor="white", markeredgewidth=1.2, label=names[p])
        trails.append(trail)
        dots.append(dot)

        if r is not None:
            # Fully transparent faces: facecolor="none" crashes 3-D depth sorting in matplotlib
            box = Poly3DCollection(_box_faces(pos[0] - r, pos[0] + r), facecolor=(0, 0, 0, 0),
                                   edgecolor=color, linewidth=0.8, linestyle="--")
            ax.add_collection3d(box)
            safety_boxes.append(box)

        boxes = []
        for _n in range(3):
            plume = Poly3DCollection(_box_faces(pos[0], pos[0] + 0.01), facecolor=THRUST_COLOR,
                                     edgecolor=THRUST_COLOR, linewidth=0.5)
            plume.set_visible(False)
            ax.add_collection3d(plume)
            boxes.append(plume)
        plume_boxes.append(boxes)

    readout = ax.text2D(0.0, 1.0, "", transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, family="monospace", color="#2b2b29")

    # ---- Axes ---------------------------------------------------------------
    lows, highs = _animation_limits(positions, cfg, r)
    ax.set_xlim3d(lows[0], highs[0])
    ax.set_ylim3d(lows[1], highs[1])
    ax.set_zlim3d(lows[2], highs[2])
    _equal_cube_axes(ax)
    labels = _axis_labels(cfg)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(_title(result, cfg), pad=30)
    ax.view_init(elev=20, azim=-20)
    handles, _labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + _legend_extras(r, cfg), frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 0.9))
    fig.subplots_adjust(left=0.0, right=0.8)    # room for the legend (tight_layout ignores 3-D well)

    # ---- One frame ----------------------------------------------------------
    def update(frame):
        now, plumes, closest, hits = _frame_state(positions, U, frame, substeps, cfg, r)
        breached = set()
        if closest is not None and closest[0] < 1.0 - 1e-6:
            breached = {closest[1], closest[2]}
        hit_plumes = {(p, k) for p, k, _q in hits}

        for p in range(V):
            trail_points = positions[p, :frame + 1]
            trails[p].set_data_3d(trail_points[:, 0], trail_points[:, 1], trail_points[:, 2])
            dots[p].set_data_3d([now[p, 0]], [now[p, 1]], [now[p, 2]])

            if r is not None:
                box = safety_boxes[p]
                box.set_verts(_box_faces(now[p] - r, now[p] + r))
                if p in breached:
                    box.set_edgecolor(WARNING_COLOR)
                    box.set_linewidth(2.5)
                else:
                    box.set_edgecolor(VEHICLE_COLORS[p])
                    box.set_linewidth(0.8)

            for k, plume in enumerate(plume_boxes[p]):
                if k >= len(plumes[p]):
                    plume.set_visible(False)
                    continue
                lower, upper, strength = plumes[p][k]
                plume.set_verts(_box_faces(lower, upper))
                plume.set_alpha(_plume_alpha(strength))
                if (p, k) in hit_plumes:
                    plume.set_edgecolor(WARNING_COLOR)
                    plume.set_linewidth(2.5)
                else:
                    plume.set_edgecolor(THRUST_COLOR)
                    plume.set_linewidth(0.5)
                plume.set_visible(True)

        readout.set_text(_readout(frame, substeps, cfg, names, closest, hits))

    anim = FuncAnimation(fig, update, frames=n_frames, interval=FRAME_INTERVAL_MS)
    _finish_animation(anim, save_path, show)
    return anim


def animate(result, cfg, save_path=None, show=True):
    """
    2-D or 3-D animation, picked from general.dims.
    If you call this with show=False, keep the returned object, or the animation stops.
    """
    if cfg["general"]["dims"] == 2:
        return animate_2d(result, cfg, save_path=save_path, show=show)
    return animate_3d(result, cfg, save_path=save_path, show=show)
