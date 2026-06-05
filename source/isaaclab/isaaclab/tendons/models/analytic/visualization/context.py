# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

td = TendonData(1, TendonConstants(), dummy_randomization)

__all__ = ["DEFAULT_ALPHA_2", "td", "tids"]
