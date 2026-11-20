"""
main.py - entry point.

Usage (run from the repo root):
    python main.py                      # everything as set in config.yaml (toggles section)
    python main.py one_obstacle
    python main.py one_obstacle --3d    # same scenario, 3-D config (config_3d.yaml)
    python main.py multi_obstacle --multi-vehicle --plumes
    python main.py no_obstacle --no-plot
    python main.py one_obstacle --config other_config.yaml
    python main.py no_obstacle --multi-vehicle --plumes --animate
    python main.py no_obstacle --multi-vehicle --plumes --gif results/plumes.gif

2-D or 3-D: set USE_3D below (handy for the VS Code run button), or pass --3d.
Animation: set ANIMATE (and optionally GIF_PATH) below, or pass --animate / --gif PATH.
--config <file> overrides both.

TODO:
    [x] argparse: optional positional scenario (no_obstacle | one_obstacle | multi_obstacle),
        --multi-vehicle, --plumes, --config (default config.yaml), --no-plot
    [x] Load and resolve the config (src.config)
    [x] Build the discrete dynamics (src.dynamics)
    [x] Build and solve the MILP (src.model)
    [x] Print a summary: scenario, toggles, fuel used, solve time, binary count
    [x] Plot the result (src.plotting)  -> needs plot_2d / plot_3d in plotting.py
    [x] --animate / --gif: vehicles moving, with safety boxes and plumes
"""

import argparse
from pathlib import Path

from src import config, dynamics, model, plotting

# ---- 2-D / 3-D toggle -------------------------------------------------------
USE_3D = False                                           # True -> run config_3d.yaml by default

CONFIG_2D = config.PATH                                   # config.yaml in the repo root
CONFIG_3D = config.PATH.parent / "config_3d.yaml"

# ---- Animation toggle -------------------------------------------------------
ANIMATE = True                # True -> show the animation after the static plot (or pass --animate)
GIF_PATH = None                 # e.g. "results/run.gif" -> also save it (or pass --gif PATH)


def parse_args():
    """
    Read the command-line options.
    """
    parser = argparse.ArgumentParser(description="MILP trajectory optimisation (Richards 2002, Sec 2.3)")

    parser.add_argument("scenario", nargs="?", default=None,
                        choices=["no_obstacle", "one_obstacle", "multi_obstacle", "u_trap"],
                        help="which obstacle scenario to run (default: toggles.scenario in config.yaml)")
    parser.add_argument("--multi-vehicle", action="store_true", default=None,
                        help="use all vehicles in config.yaml (default: toggles.multi_vehicle)")
    parser.add_argument("--plumes", action="store_true", default=None,
                        help="add plume-impingement constraints (needs --multi-vehicle)")
    parser.add_argument("--3d", dest="use_3d", action="store_true",
                        help="use config_3d.yaml instead of config.yaml")
    parser.add_argument("--config", default=None,
                        help="path to any config file (overrides --3d and USE_3D)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the plot (and the animation)")
    parser.add_argument("--animate", action="store_true",
                        help="after the static plot, show the vehicles moving (safety boxes, plumes)")
    parser.add_argument("--gif", default=None, metavar="PATH",
                        help="save the animation as a .gif, e.g. results/run.gif (implies --animate)")

    return parser.parse_args()


def pick_config_path(args):
    """Which config file to use: --config first, then --3d / USE_3D, else the 2-D config."""
    if args.config is not None:
        return args.config
    if args.use_3d or USE_3D:
        return CONFIG_3D
    return CONFIG_2D


def print_summary(cfg, result, config_path):
    """Print what was run and the key results."""
    print()
    print("=" * 50)
    print(f"Config:            {Path(config_path).name}  ({cfg['general']['dims']}-D, {cfg['general']['dynamics']})")
    print(f"Scenario:          {cfg['scenario']}  ({len(cfg['obstacles'])} obstacle(s))")
    print(f"Vehicles:          {len(cfg['vehicles'])}")
    print(f"Vehicle avoidance: {cfg['vehicle_avoidance']}")
    print(f"Plumes:            {cfg['plumes']}")
    print("-" * 50)
    print(f"Status:            {result['status']}")
    print(f"Fuel (sum |u|):    {result['fuel']:.6g} N")
    print(f"Total delta-v:     {result['delta_v']:.6g} m/s")
    print(f"Solve time:        {result['runtime']:.3f} s")
    print(f"Binary variables:  {result['n_binaries']}")
    print(f"MIP gap:           {100 * result['mip_gap']:.2f} %")
    print("=" * 50)


def main():
    args = parse_args()

    # 1. Config: pick the file (2-D or 3-D), apply the command-line toggles, check, resolve
    config_path = pick_config_path(args)
    cfg = config.get_config(config_path, args.scenario, args.multi_vehicle, args.plumes)

    # 2. Dynamics: discrete (Ad, Bd) for the chosen model
    Ad, Bd = dynamics.get_discrete_model(cfg)

    # 3. Build and solve the MILP
    result = model.build_and_solve(cfg, Ad, Bd)

    # 4. Summary
    print_summary(cfg, result, config_path)

    # 5. Plot
    if not args.no_plot:
        if cfg["general"]["dims"] == 2:
            plotting.plot_2d(result, cfg)
        else:
            plotting.plot_3d(result, cfg)

    # 6. Animation (opens after the static plot window is closed)
    #    The command line wins over the constants at the top of this file
    gif_path = args.gif if args.gif is not None else GIF_PATH
    animate = args.animate or ANIMATE or gif_path is not None

    if not args.no_plot and animate:
        plotting.animate(result, cfg, save_path=gif_path)


if __name__ == "__main__":
    main()
