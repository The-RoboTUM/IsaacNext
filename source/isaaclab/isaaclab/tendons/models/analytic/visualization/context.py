"""Shared visualization constants/data.

This mirrors the original drawing script: the visualizer uses the same baseline
analytic tendon constants and a one-sample TendonData object for geometry.
"""

import numpy as np

from isaaclab.tendons.models.analytic.constants import TendonConstants, dummy_randomization, tids
from isaaclab.tendons.models.analytic.tendon_data import TendonData

# draw the leg, starting at joint 2
DEFAULT_ALPHA_2 = np.deg2rad(300)

# Keep these module-level objects so the split code behaves like the original script.
tc = TendonConstants()
td = TendonData(1, dummy_randomization)

__all__ = ["DEFAULT_ALPHA_2", "tc", "td", "tids"]
