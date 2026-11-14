"""
config.py - load config.yaml, apply command-line toggles, run sanity checks.

TODO:
    [x] load_config(path): read config.yaml into a dict
    [x] apply_toggles(cfg, args): override toggles.scenario / multi_vehicle / plumes
        with the command-line values
    [x] Toggle rules:
        - multi_vehicle false -> keep only vehicles[0], turn vehicle avoidance off
        - multi_vehicle false and plumes true -> warn and set plumes to false
        - multi_vehicle true  -> use all vehicles, turn vehicle avoidance on
    [x] Look up the chosen scenario's obstacle list
    [x] Sanity checks (only for mistakes that would otherwise run silently with a
        wrong result; mistakes that crash anyway, or that Gurobi reports as
        infeasible, are left to Python/Gurobi):
        - dynamics / dims are valid options, numbers have the right sign
        - x_max, start/end and obstacle corners have the right length
        - lower < upper for every obstacle
        - start/end positions lie outside every inflated obstacle
        - toggles are true/false, multi_vehicle requires at least 2 vehicles
    [x] Return one resolved config object for model.py / plotting.py
    [ ] Later: check vehicle_avoidance.safety_distance and plume.length/half_width
        once the vehicle and plume constraints exist

Usage:
    from src.config import get_config
    cfg = get_config(scenario="one_obstacle")
"""

import warnings
from pathlib import Path

import yaml

# Default config file: config.yaml in the repo root (one folder up from src/)
PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# =============================================================================
# 1. Loading
# =============================================================================

def load_configuration(path=PATH):
    """Read the YAML file and return it as a dict."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg


# =============================================================================
# 2. Toggles
# =============================================================================

def apply_toggles(cfg, scenario=None, multi_vehicle=None, plumes=None):
    """Overwrite the toggles from config.yaml with any value that is given (not None)."""
    toggles = cfg["toggles"]

    if scenario is not None:
        toggles["scenario"] = scenario
    if multi_vehicle is not None:
        toggles["multi_vehicle"] = multi_vehicle
    if plumes is not None:
        toggles["plumes"] = plumes

    return cfg


# =============================================================================
# 3. Sanity checks
#    Only mistakes that would otherwise run without crashing and give a wrong result.
# =============================================================================

def _is_inside(point, lower, upper, margin):
    """True if the point is strictly inside the box grown by `margin` on every side."""
    for n in range(len(point)):
        if point[n] <= lower[n] - margin:
            return False
        if point[n] >= upper[n] + margin:
            return False
    return True


def _check_general(general):
    """Check the 'general' section: valid options and signs."""
    if general["dynamics"] not in ("double_integrator", "hill"):
        raise ValueError(f"general.dynamics must be 'double_integrator' or 'hill', "
                         f"got {general['dynamics']!r}")

    if general["dims"] not in (2, 3):
        raise ValueError(f"general.dims must be 2 or 3, got {general['dims']!r}")

    # A zero or negative value would still solve, but give a meaningless result
    positive_keys = ["T", "dt", "mass", "u_max", "big_M"]
    if general["dynamics"] == "hill":
        positive_keys.append("omega")

    for key in positive_keys:
        if general[key] <= 0:
            raise ValueError(f"general.{key} must be positive, got {general[key]!r}")

    if general.get("obstacle_margin", 0.0) < 0:
        raise ValueError(f"general.obstacle_margin must be >= 0, got {general['obstacle_margin']!r}")

    # One state limit per state component: [positions..., velocities...]
    n_state = 2 * general["dims"]
    x_max = general["x_max"]
    if len(x_max) != n_state:
        raise ValueError(f"general.x_max must have {n_state} values, got {x_max!r}")
    for value in x_max:
        if value <= 0:
            raise ValueError(f"general.x_max values must all be positive, got {x_max!r}")


def _check_vehicles(vehicles, general):
    """Check each vehicle's start and end state has one value per state component."""
    if len(vehicles) == 0:
        raise ValueError("vehicles must contain at least one vehicle")

    n_state = 2 * general["dims"]

    for i, vehicle in enumerate(vehicles):
        name = vehicle.get("name", f"vehicle {i}")
        for key in ("start", "end"):
            if len(vehicle[key]) != n_state:
                raise ValueError(f"{name}.{key} must have {n_state} values, got {vehicle[key]!r}")


def _check_obstacles(scenarios, vehicles, general):
    """
    Check the obstacles in every scenario (not only the selected one),
    so a broken scenario is caught before you switch to it.
    """
    dims = general["dims"]
    margin = general.get("obstacle_margin", 0.0)

    for scenario_name, obstacles in scenarios.items():
        if obstacles is None:      # a blank entry in YAML means no obstacles
            continue

        for j, obstacle in enumerate(obstacles):
            where = f"scenarios.{scenario_name}[{j}]"
            lower = obstacle["lower"]
            upper = obstacle["upper"]

            # Corner vectors have one value per axis
            if len(lower) != dims or len(upper) != dims:
                raise ValueError(f"{where}: lower and upper must each have {dims} values")

            # A box with lower >= upper on any axis is empty, i.e. no obstacle at all
            for n in range(dims):
                if lower[n] >= upper[n]:
                    raise ValueError(f"{where}: lower[{n}] = {lower[n]} must be smaller "
                                     f"than upper[{n}] = {upper[n]}")

            # Avoidance is only enforced from step 1 to T-1, so a start inside a box
            # would not be caught by the solver
            for i, vehicle in enumerate(vehicles):
                name = vehicle.get("name", f"vehicle {i}")
                for key in ("start", "end"):
                    position = vehicle[key][:dims]
                    if _is_inside(position, lower, upper, margin):
                        raise ValueError(f"{name}.{key} position {position} is inside obstacle "
                                         f"{where} (inflated by margin {margin})")


def _check_toggles(toggles, vehicles):
    """Check the toggle values and that they fit the rest of the config."""
    # A string like "false" would count as True in an if-statement
    for key in ("multi_vehicle", "plumes"):
        if not isinstance(toggles[key], bool):
            raise ValueError(f"toggles.{key} must be true or false, got {toggles[key]!r}")

    if toggles["multi_vehicle"] and len(vehicles) < 2:
        raise ValueError("multi_vehicle is on, but config.yaml defines fewer than 2 vehicles")


def validate(cfg):
    """Raise ValueError on the first problem found. Returns nothing if all is fine."""
    general = cfg["general"]
    vehicles = cfg["vehicles"]

    # Order matters: later checks rely on the earlier ones having passed
    _check_general(general)
    _check_vehicles(vehicles, general)
    _check_obstacles(cfg["scenarios"], vehicles, general)
    _check_toggles(cfg["toggles"], vehicles)


# =============================================================================
# 4. Resolve: apply the toggle rules and build what model.py / plotting.py use
# =============================================================================

def resolve(cfg):
    """
    Apply the toggle rules and return one flat dict with only what this run needs.

    Toggle rules:
        multi_vehicle off -> only the first vehicle, no vehicle avoidance, no plumes
        multi_vehicle on  -> all vehicles, vehicle avoidance on
        plumes            -> only allowed with multi_vehicle on
    """
    toggles = cfg["toggles"]
    scenario = toggles["scenario"]
    multi_vehicle = toggles["multi_vehicle"]
    plumes = toggles["plumes"]

    # Obstacles for the chosen scenario (a blank YAML entry gives None -> no obstacles)
    obstacles = cfg["scenarios"][scenario]
    if obstacles is None:
        obstacles = []

    # Vehicles and the constraints that depend on them
    if multi_vehicle:
        vehicles = cfg["vehicles"]
        vehicle_avoidance = True
    else:
        vehicles = cfg["vehicles"][:1]
        vehicle_avoidance = False
        if plumes:
            warnings.warn("plumes is on but multi_vehicle is off: plumes are ignored for one vehicle")
            plumes = False

    # Optional sections, only used once vehicle avoidance / plumes are implemented
    safety_distance = cfg.get("vehicle_avoidance", {}).get("safety_distance")
    plume = cfg.get("plume", {})

    return {
        "general": cfg["general"],
        "scenario": scenario,
        "obstacles": obstacles,
        "vehicles": vehicles,
        "vehicle_avoidance": vehicle_avoidance,
        "plumes": plumes,
        "safety_distance": safety_distance,
        "plume": plume,
    }


# =============================================================================
# 5. One call for main.py
# =============================================================================

def get_config(path=PATH, scenario=None, multi_vehicle=None, plumes=None):
    """Load -> apply command-line toggles -> validate -> resolve."""
    cfg = load_configuration(path)
    cfg = apply_toggles(cfg, scenario, multi_vehicle, plumes)
    validate(cfg)
    return resolve(cfg)
