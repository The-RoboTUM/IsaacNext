# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared visualization constants/data.

This mirrors the original drawing script: the visualizer uses the same baseline
analytic tendon constants and a one-sample TendonData object for geometry.
"""

import numpy as np

from isaaclab.tendons.models.analytic.constants import tids
from isaaclab.tendons.models.analytic.tendon_data import TendonData
from isaaclab.tendons.parameter_loader import load_forrest_parameter_config

# Draw the leg from joint 2 in the +X/Z sagittal projection. The new Forrest
# URDF has j2->j3 at atan2(z, x) ~= -80 deg.
DEFAULT_ALPHA_2 = np.deg2rad(280)

# Keep these module-level objects so the split code behaves like the original script.
_forrest_params = load_forrest_parameter_config()
tc = _forrest_params.to_tendon_constants()
_randomization = _forrest_params.to_tendon_randomization_ranges()
td = TendonData(1, _randomization, tc=tc)

__all__ = ["DEFAULT_ALPHA_2", "tc", "td", "tids"]
