# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# isaaclab/tendons/__init__.py

__all__ = [
    "TendonManager",
    "AnalyticTendonEnergyModel",
]


def __getattr__(name):
    if name == "TendonManager":
        from isaaclab.tendons.manager import TendonManager

        return TendonManager

    if name == "AnalyticTendonEnergyModel":
        from isaaclab.tendons.models.analytic.analytic_energy_model import AnalyticTendonEnergyModel

        return AnalyticTendonEnergyModel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
