# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime calibration helpers for the standalone Forrest tendon runner."""

__all__ = [
    "CalibrationState",
    "CalibrationWindows",
    "ForrestTendonOverlay",
    "apply_tendon_parameters",
    "build_calibration_state",
    "build_tendon_data_from_state",
    "runtime_controller_command_tensor",
]


def __getattr__(name: str):
    if name == "runtime_controller_command_tensor":
        from .controller import runtime_controller_command_tensor

        return runtime_controller_command_tensor
    if name in {
        "CalibrationState",
        "apply_tendon_parameters",
        "build_calibration_state",
        "build_tendon_data_from_state",
    }:
        from . import state

        return getattr(state, name)
    if name == "ForrestTendonOverlay":
        from .tendon_overlay import ForrestTendonOverlay

        return ForrestTendonOverlay
    if name == "CalibrationWindows":
        from .ui import CalibrationWindows

        return CalibrationWindows
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
