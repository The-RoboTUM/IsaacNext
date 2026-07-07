# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime controller evaluation for live calibration."""

from __future__ import annotations

import torch

from isaaclab.tendons.calibration.state import CalibrationState
from isaaclab.tendons.controllers.cpg import BirdBotCPGLeg, CPGParams, HopfCPGLeg, HopfCPGParams
from isaaclab.tendons.controllers.sinusoidal import SinusoidalLegController, SinusoidalParams
from isaaclab.tendons.runner import ActuatedDofSpec


def _controller_pair(controller: str, values: dict[str, float], actuated_dof_specs: list[ActuatedDofSpec]):
    if controller == "cpg":
        common = dict(
            f_hz=values["run.cpg.f_hz"],
            D=values["run.cpg.duty_factor"],
            A_h_deg=values["run.cpg.hip_amplitude_deg"],
            O_h_deg=values["run.cpg.hip_offset_deg"],
            A_k_deg=values["run.cpg.knee_amplitude_deg"],
            S_f=values["run.cpg.swing_start_offset"],
            S_e=values["run.cpg.swing_end_offset"],
        )
        combined_phase = values["run.cpg.combined_phase_offset_rad"]
        include_knee = bool(round(values.get("run.cpg.include_knee", 1.0)))
        left = BirdBotCPGLeg(
            CPGParams(phi0=values["run.cpg.left_phase_offset_rad"] + combined_phase, **common),
            include_knee=include_knee,
        )
        right = BirdBotCPGLeg(
            CPGParams(phi0=values["run.cpg.right_phase_offset_rad"] + combined_phase, **common),
            include_knee=include_knee,
        )
        return left, right
    if controller == "cpg_oscillator":
        common = dict(
            f_hz=values["run.cpg_oscillator.f_hz"],
            duty_factor=values["run.cpg_oscillator.duty_factor"],
            hip_flexion_amplitude_deg=values["run.cpg_oscillator.hip_flexion_amplitude_deg"],
            hip_flexion_offset_deg=values["run.cpg_oscillator.hip_flexion_offset_deg"],
            hip_flexion_phase_rad=values["run.cpg_oscillator.hip_flexion_phase_rad"],
            knee_flexion_amplitude_deg=values["run.cpg_oscillator.knee_flexion_amplitude_deg"],
            knee_flexion_offset_deg=values["run.cpg_oscillator.knee_flexion_offset_deg"],
            knee_flexion_phase_rad=values["run.cpg_oscillator.knee_flexion_phase_rad"],
            knee_swing_power=values["run.cpg_oscillator.knee_swing_power"],
            hip_roll_amplitude_deg=values["run.cpg_oscillator.hip_roll_amplitude_deg"],
            hip_roll_offset_deg=values["run.cpg_oscillator.hip_roll_offset_deg"],
            hip_roll_phase_rad=values["run.cpg_oscillator.hip_roll_phase_rad"],
            hip_yaw_amplitude_deg=values["run.cpg_oscillator.hip_yaw_amplitude_deg"],
            hip_yaw_offset_deg=values["run.cpg_oscillator.hip_yaw_offset_deg"],
            hip_yaw_phase_rad=values["run.cpg_oscillator.hip_yaw_phase_rad"],
        )
        return (
            HopfCPGLeg(HopfCPGParams(phi0=values["run.cpg_oscillator.left_phase_rad"], **common)),
            HopfCPGLeg(HopfCPGParams(phi0=values["run.cpg_oscillator.right_phase_rad"], **common)),
        )
    if controller == "sin":
        dofs = {spec.dof for spec in actuated_dof_specs}
        common = dict(
            f_hz=values["run.sinusoidal.f_hz"],
            amplitude_deg={dof: values.get(f"run.sinusoidal.amplitude_deg.{dof}", 0.0) for dof in dofs},
            offset_deg={dof: values.get(f"run.sinusoidal.offset_deg.{dof}", 0.0) for dof in dofs},
        )
        return (
            SinusoidalLegController(
                SinusoidalParams(
                    phi0=values["run.sinusoidal.left_phi0_rad"],
                    phase_rad={dof: values.get(f"run.sinusoidal.left_phase_rad.{dof}", 0.0) for dof in dofs},
                    **common,
                )
            ),
            SinusoidalLegController(
                SinusoidalParams(
                    phi0=values["run.sinusoidal.right_phi0_rad"],
                    phase_rad={dof: values.get(f"run.sinusoidal.right_phase_rad.{dof}", 0.0) for dof in dofs},
                    **common,
                )
            ),
        )
    raise ValueError(f"Unknown controller: {controller!r}")


def runtime_controller_command_tensor(
    *,
    t: float,
    state: CalibrationState,
    actuated_dof_specs: list[ActuatedDofSpec],
    initial_joint_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return live calibrated joint targets and signed controller command deltas."""

    values = state.snapshot_values()
    controller = state.get_controller()
    left_controller, right_controller = _controller_pair(controller, values, actuated_dof_specs)
    controllers = {"left": left_controller, "right": right_controller}
    commands = []
    for spec in actuated_dof_specs:
        q, _qd = controllers[spec.side].joint(spec.dof, t)
        commands.append(spec.sign * q)
    command_delta = torch.tensor(
        [commands],
        dtype=initial_joint_positions.dtype,
        device=initial_joint_positions.device,
    )
    return initial_joint_positions + command_delta, command_delta
