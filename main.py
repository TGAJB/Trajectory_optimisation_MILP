"""
main.py - entry point.

Usage (run from the repo root):
    python main.py                      # everything as set in config.yaml (toggles section)
    python main.py one_obstacle
    python main.py multi_obstacle --multi-vehicle --plumes
    python main.py no_obstacle --no-plot
    python main.py one_obstacle --config other_config.yaml

TODO:
    [x] argparse: optional positional scenario (no_obstacle | one_obstacle | multi_obstacle),
        --multi-vehicle, --plumes, --config (default config.yaml), --no-plot
    [x] Load and resolve the config (src.config)
    [x] Build the discrete dynamics (src.dynamics)
    [x] Build and solve the MILP (src.model)
    [x] Print a summary: scenario, toggles, fuel used, solve time, binary count
    [x] Plot the result (src.plotting)  -> needs plot_2d / plot_3d in plotting.py
"""

import argparse

from src import config, dynamics, model, plotting


def parse_args():
    """
    Read the command-line options.
    """
    parser = argparse.ArgumentParser(description="MILP trajectory optimisation (Richards 2002, Sec 2.3)")

    parser.add_argument("scenario", nargs="?", default=None,
                        choices=["no_obstacle", "one_obstacle", "multi_obstacle"],
                        help="which obstacle scenario to run (default: toggles.scenario in config.yaml)")
    parser.add_argument("--multi-vehicle", action="store_true", default=None,
                        help="use all vehicles in config.yaml (default: toggles.multi_vehicle)")
    parser.add_argument("--plumes", action="store_true", default=None,
                        help="add plume-impingement constraints (needs --multi-vehicle)")
    parser.add_argument("--config", default=config.PATH,
                        help="path to the config file (default: config.yaml in the repo root)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the plot")

    return parser.parse_args()


def print_summary(cfg, result):
    """Print what was run and the key results."""
    print()
    print("=" * 50)
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

    # 1. Config: load config.yaml, apply the command-line toggles, check, resolve
    cfg = config.get_config(args.config, args.scenario, args.multi_vehicle, args.plumes)

    # 2. Dynamics: discrete (Ad, Bd) for the chosen model
    Ad, Bd = dynamics.get_discrete_model(cfg)

    # 3. Build and solve the MILP
    result = model.build_and_solve(cfg, Ad, Bd)

    # 4. Summary
    print_summary(cfg, result)

    # 5. Plot
    if not args.no_plot:
        if cfg["general"]["dims"] == 2:
            plotting.plot_2d(result, cfg)
        else:
            plotting.plot_3d(result, cfg)


if __name__ == "__main__":
    main()
