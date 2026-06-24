# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TorchScript kernels used by the PSO runner.

The Isaac simulation loop remains Python-driven, but the dense per-env tensor
math lives here so it can be scripted and reused without per-step dictionary
dispatch overhead.
"""

from __future__ import annotations

import torch


@torch.jit.script
def pso_integrate_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    personal_best_positions: torch.Tensor,
    social_best_positions: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    inertia: float,
    cognitive: float,
    social: float,
    velocity_clamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance bounded normalized PSO positions by one velocity update."""

    new_velocities = (
        inertia * velocities
        + cognitive * r1 * (personal_best_positions - positions)
        + social * r2 * (social_best_positions - positions)
    )
    if velocity_clamp > 0.0:
        new_velocities = new_velocities.clamp(-velocity_clamp, velocity_clamp)

    new_positions = positions + new_velocities
    out_of_bounds = (new_positions < 0.0) | (new_positions > 1.0)
    new_positions = new_positions.clamp(0.0, 1.0)
    new_velocities = torch.where(out_of_bounds, -0.5 * new_velocities, new_velocities)
    return new_positions, new_velocities


@torch.jit.script
def ring_best_positions(
    personal_best_positions: torch.Tensor,
    personal_best_scores: torch.Tensor,
    neighborhood_size: int,
) -> torch.Tensor:
    """Return each particle's best local-neighborhood position on a ring."""

    radius = max(0, neighborhood_size // 2)
    best_scores = personal_best_scores.clone()
    best_positions = personal_best_positions.clone()
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        candidate_scores = torch.roll(personal_best_scores, shifts=offset, dims=0)
        candidate_positions = torch.roll(personal_best_positions, shifts=offset, dims=0)
        improved = candidate_scores > best_scores
        best_scores = torch.where(improved, candidate_scores, best_scores)
        best_positions = torch.where(improved.unsqueeze(1), candidate_positions, best_positions)
    return best_positions


@torch.jit.script
def cpg_oscillator_command_kernel(
    t: torch.Tensor,
    initial_joint_positions: torch.Tensor,
    controller_zero: torch.Tensor,
    joint_side_ids: torch.Tensor,
    joint_dof_ids: torch.Tensor,
    joint_signs: torch.Tensor,
    f_hz: torch.Tensor,
    duty_factor: torch.Tensor,
    left_phase_rad: torch.Tensor,
    right_phase_rad: torch.Tensor,
    hip_flexion_amplitude_rad: torch.Tensor,
    hip_flexion_offset_rad: torch.Tensor,
    hip_flexion_phase_rad: torch.Tensor,
    knee_flexion_amplitude_rad: torch.Tensor,
    knee_flexion_offset_rad: torch.Tensor,
    knee_flexion_phase_rad: torch.Tensor,
    knee_swing_power: torch.Tensor,
    hip_roll_amplitude_rad: torch.Tensor,
    hip_roll_offset_rad: torch.Tensor,
    hip_roll_phase_rad: torch.Tensor,
    hip_yaw_amplitude_rad: torch.Tensor,
    hip_yaw_offset_rad: torch.Tensor,
    hip_yaw_phase_rad: torch.Tensor,
) -> torch.Tensor:
    """Vectorized phase-warped CPG joint target kernel."""

    two_pi = 2.0 * torch.pi
    base_phase = two_pi * f_hz * t
    duty = duty_factor.clamp(0.05, 0.95)
    knee_power = knee_swing_power.clamp(0.1, 6.0)
    commands = torch.empty_like(initial_joint_positions)

    num_joints = int(joint_dof_ids.shape[0])
    for joint_index in range(num_joints):
        phase_offset = torch.where(joint_side_ids[joint_index] == 0, left_phase_rad, right_phase_rad)
        phase = torch.remainder(base_phase + phase_offset, two_pi)
        stance_phase = phase / (2.0 * duty)
        swing_phase = phase / (2.0 * (1.0 - duty)) + torch.pi * (1.0 - 2.0 * duty) / (1.0 - duty)
        theta = torch.where(phase <= two_pi * duty, stance_phase, swing_phase)

        dof_id = int(joint_dof_ids[joint_index])
        if dof_id == 0:
            q = hip_roll_amplitude_rad * torch.sin(theta + hip_roll_phase_rad) + hip_roll_offset_rad
        elif dof_id == 1:
            q = hip_yaw_amplitude_rad * torch.sin(theta + hip_yaw_phase_rad) + hip_yaw_offset_rad
        elif dof_id == 2:
            q = hip_flexion_amplitude_rad * torch.sin(theta + hip_flexion_phase_rad) + hip_flexion_offset_rad
        else:
            swing = torch.clamp(torch.sin(theta + knee_flexion_phase_rad), min=0.0)
            q = knee_flexion_offset_rad + knee_flexion_amplitude_rad * torch.pow(swing, knee_power)
        commands[:, joint_index] = joint_signs[joint_index] * q

    return initial_joint_positions + commands - controller_zero
