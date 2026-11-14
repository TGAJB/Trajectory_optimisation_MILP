"""
config.py - load config.yaml, apply command-line toggles, run sanity checks.

TODO:
    [ ] load_config(path): read config.yaml into a dict
    [ ] apply_toggles(cfg, args): override toggles.scenario / multi_vehicle / plumes
        with the command-line values
    [ ] Toggle rules:
        - multi_vehicle false -> keep only vehicles[0], turn vehicle avoidance off
        - multi_vehicle false and plumes true -> warn and set plumes to false
        - multi_vehicle true  -> use all vehicles, turn vehicle avoidance on
    [ ] Look up the chosen scenario's obstacle list (error if the name is unknown)
    [ ] Sanity checks:
        - required general fields are not null
        - start/end state lengths equal 2 * dims
        - obstacle lower/upper lengths equal dims, and lower < upper
        - start/end positions lie outside every inflated obstacle
        - multi_vehicle requires at least 2 vehicles
    [ ] Return one resolved config object for model.py / plotting.py
"""

import warnings
from pathlib import Path

import yaml
