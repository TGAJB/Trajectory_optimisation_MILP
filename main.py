"""
main.py - entry point.

Usage (planned):
    python main.py one_obstacle
    python main.py multi_obstacle --multi-vehicle --plumes

TODO:
    [ ] argparse: optional positional scenario (no_obstacle | one_obstacle | multi_obstacle),
        --multi-vehicle, --plumes, --config (default config.yaml), --no-plot
    [ ] Load and resolve the config (src.config)
    [ ] Build the discrete dynamics (src.dynamics)
    [ ] Build and solve the MILP (src.model)
    [ ] Print a summary: scenario, toggles, fuel used, solve time, binary count
    [ ] Plot the result (src.plotting)
"""

import argparse

from src import config, dynamics, model, plotting
