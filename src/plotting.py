"""
plotting.py - visualise trajectories and obstacles.

TODO:
    [ ] plot_2d(result, cfg): trajectory per vehicle, obstacles as rectangles
        (actual and inflated), start/end markers
    [ ] plot_3d(result, cfg): same in 3-D with boxes drawn as Poly3DCollection
    [ ] Thrust arrows at time steps where the vehicle fires
    [ ] Later: draw the safety box around vehicles and the plume regions
    [ ] Optionally save the figure to results/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
