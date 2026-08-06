# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamics term extraction for small-articulation audits."""

from __future__ import annotations

from typing import Any

import torch

BASE_COORDINATE_COUNT = 6


def compute_articulation_dynamics_terms(
    robot,
    *,
    dt: float,
    command_effort: torch.Tensor,
    previous_joint_vel: torch.Tensor | None = None,
    previous_command_effort: torch.Tensor | None = None,
    include_base: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute generalized dynamics terms for an IsaacLab articulation.

    The returned tensors are joint-only by default. Passing ``include_base=True``
    keeps floating-base slots when the PhysX API exposes them.
    """

    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    joint_acc = robot.data.joint_acc
    width = _generalized_width(robot, include_base=include_base)
    mass_matrix = _generalized_mass_matrix(robot, include_base=include_base)
    generalized_acc = _generalized_acceleration(robot, joint_acc=joint_acc, include_base=include_base)
    inertia = torch.bmm(mass_matrix, generalized_acc.unsqueeze(-1)).squeeze(-1)

    if previous_joint_vel is None:
        joint_acc_fd = torch.zeros_like(joint_acc)
    else:
        joint_acc_fd = (joint_vel - previous_joint_vel) / float(dt)
    generalized_acc_fd = _generalized_acceleration(robot, joint_acc=joint_acc_fd, include_base=include_base)
    inertia_fd = torch.bmm(mass_matrix, generalized_acc_fd.unsqueeze(-1)).squeeze(-1)

    gravity_force_api = _generalized_force_or_zero(
        robot,
        "get_generalized_gravity_forces",
        width=width,
        include_base=include_base,
    )
    gravity_compensation = _generalized_force_or_zero(
        robot,
        "get_gravity_compensation_forces",
        width=width,
        include_base=include_base,
    )
    coriolis_force_api = _generalized_force_or_zero(
        robot,
        "get_coriolis_and_centrifugal_forces",
        width=width,
        include_base=include_base,
    )
    coriolis_compensation = _generalized_force_or_zero(
        robot,
        "get_coriolis_and_centrifugal_compensation_forces",
        width=width,
        include_base=include_base,
    )

    command_effort = _normalize_generalized_vector(robot, command_effort, width=width, include_base=include_base)
    previous_command_effort = (
        torch.zeros_like(command_effort)
        if previous_command_effort is None
        else _normalize_generalized_vector(robot, previous_command_effort, width=width, include_base=include_base)
    )
    applied_torque = _normalize_generalized_vector(
        robot,
        _robot_data_or_zero(robot, "applied_torque"),
        width=width,
        include_base=include_base,
    )
    computed_torque = _normalize_generalized_vector(
        robot,
        _robot_data_or_zero(robot, "computed_torque"),
        width=width,
        include_base=include_base,
    )
    joint_effort_target = _normalize_generalized_vector(
        robot,
        _robot_data_or_zero(robot, "joint_effort_target"),
        width=width,
        include_base=include_base,
    )
    implicit_drive_estimate_joint, implicit_drive_saturation_joint = _implicit_drive_estimate(robot)
    implicit_drive_estimate = _normalize_generalized_vector(
        robot,
        implicit_drive_estimate_joint,
        width=width,
        include_base=include_base,
    )
    implicit_drive_saturation = _normalize_generalized_vector(
        robot,
        implicit_drive_saturation_joint,
        width=width,
        include_base=include_base,
    )

    physx_actuation = _generalized_force_or_zero(
        robot,
        "get_dof_actuation_forces",
        width=width,
        include_base=include_base,
    )
    solver_joint = _generalized_force_or_zero(
        robot,
        "get_dof_projected_joint_forces",
        width=width,
        include_base=include_base,
    )
    zero = joint_pos.new_zeros((joint_pos.shape[0], width))

    selected_conservative = (
        inertia
        - gravity_compensation_actual(gravity_compensation)
        - coriolis_compensation_actual(coriolis_compensation)
    )
    selected_residual = selected_conservative - command_effort

    return {
        "q": _normalize_generalized_vector(robot, joint_pos, width=width, include_base=include_base),
        "dq": _normalize_generalized_vector(robot, joint_vel, width=width, include_base=include_base),
        "ddq": generalized_acc,
        "ddq_fd": generalized_acc_fd,
        "mass_matrix": mass_matrix,
        "inertia": inertia,
        "inertia_fd": inertia_fd,
        "gravity_force_api": gravity_force_api,
        "gravity_compensation": gravity_compensation,
        "gravity_compensation_actual": gravity_compensation_actual(gravity_compensation),
        "coriolis_force_api": coriolis_force_api,
        "coriolis_compensation": coriolis_compensation,
        "coriolis_compensation_actual": coriolis_compensation_actual(coriolis_compensation),
        "actuation_command": command_effort,
        "actuation_previous_command": previous_command_effort,
        "applied_torque": applied_torque,
        "computed_torque": computed_torque,
        "joint_effort_target": joint_effort_target,
        "implicit_drive_estimate": implicit_drive_estimate,
        "implicit_drive_saturation": implicit_drive_saturation,
        "physx_actuation": physx_actuation,
        "solver_joint": solver_joint,
        "contact": zero,
        "friction": zero,
        "residual_selected": selected_residual,
    }


def gravity_compensation_actual(gravity_compensation: torch.Tensor) -> torch.Tensor:
    """Convert PhysX compensation gravity to the opposite generalized-force alias."""

    return -gravity_compensation


def coriolis_compensation_actual(coriolis_compensation: torch.Tensor) -> torch.Tensor:
    """Convert PhysX compensation Coriolis/bias to the opposite generalized-force alias."""

    return -coriolis_compensation


def _generalized_width(robot, *, include_base: bool) -> int:
    joint_count = int(robot.data.joint_pos.shape[1])
    if include_base and not bool(getattr(robot, "is_fixed_base", True)):
        return BASE_COORDINATE_COUNT + joint_count
    return joint_count


def _generalized_mass_matrix(robot, *, include_base: bool) -> torch.Tensor:
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()
    joint_count = int(robot.data.joint_pos.shape[1])
    width = _generalized_width(robot, include_base=include_base)
    if mass_matrix.shape[-1] == width:
        return mass_matrix
    if mass_matrix.shape[-1] == joint_count and width == joint_count:
        return mass_matrix
    if mass_matrix.shape[-1] == BASE_COORDINATE_COUNT + joint_count and width == joint_count:
        return mass_matrix[:, BASE_COORDINATE_COUNT:, BASE_COORDINATE_COUNT:]
    if mass_matrix.shape[-1] == joint_count and width == BASE_COORDINATE_COUNT + joint_count:
        full = mass_matrix.new_zeros((mass_matrix.shape[0], width, width))
        full[:, BASE_COORDINATE_COUNT:, BASE_COORDINATE_COUNT:] = mass_matrix
        return full
    raise RuntimeError(f"Unexpected generalized mass matrix shape: {tuple(mass_matrix.shape)}")


def _generalized_acceleration(robot, *, joint_acc: torch.Tensor, include_base: bool) -> torch.Tensor:
    if _generalized_width(robot, include_base=include_base) == joint_acc.shape[-1]:
        return joint_acc
    root_acc = robot.root_physx_view.get_link_accelerations()[:, 0, :]
    return torch.cat((root_acc, joint_acc), dim=-1)


def _generalized_force_or_zero(
    robot,
    api_name: str,
    *,
    width: int,
    include_base: bool,
) -> torch.Tensor:
    try:
        values = getattr(robot.root_physx_view, api_name)()
    except Exception:
        return robot.data.joint_pos.new_zeros((robot.data.joint_pos.shape[0], width))
    return _normalize_generalized_vector(robot, values, width=width, include_base=include_base)


def _normalize_generalized_vector(
    robot,
    values: torch.Tensor,
    *,
    width: int,
    include_base: bool,
) -> torch.Tensor:
    values = values.to(device=robot.device)
    joint_count = int(robot.data.joint_pos.shape[1])
    full_count = BASE_COORDINATE_COUNT + joint_count
    if values.shape[-1] == width:
        return values
    if values.shape[-1] == joint_count and width == full_count:
        base = values.new_zeros((values.shape[0], BASE_COORDINATE_COUNT))
        return torch.cat((base, values), dim=-1)
    if values.shape[-1] == full_count and width == joint_count:
        return values[:, BASE_COORDINATE_COUNT:]
    if values.shape[-1] == joint_count and width == joint_count:
        return values
    if not include_base and values.shape[-1] == full_count:
        return values[:, BASE_COORDINATE_COUNT:]
    raise RuntimeError(f"Unexpected generalized vector shape {tuple(values.shape)} for width {width}.")


def _robot_data_or_zero(robot, name: str) -> torch.Tensor:
    values = getattr(robot.data, name, None)
    if values is None:
        return torch.zeros_like(robot.data.joint_pos)
    return values


def _implicit_drive_estimate(robot) -> tuple[torch.Tensor, torch.Tensor]:
    stiffness = _robot_data_or_zero(robot, "joint_stiffness")
    damping = _robot_data_or_zero(robot, "joint_damping")
    effort_target = _robot_data_or_zero(robot, "joint_effort_target")
    position_target = getattr(robot.data, "joint_pos_target", robot.data.joint_pos)
    velocity_target = getattr(robot.data, "joint_vel_target", torch.zeros_like(robot.data.joint_vel))
    computed = stiffness * (position_target - robot.data.joint_pos)
    computed = computed + damping * (velocity_target - robot.data.joint_vel) + effort_target
    effort_limits = getattr(robot.data, "joint_effort_limits", None)
    if effort_limits is None:
        effort_limits = torch.full_like(robot.data.joint_pos, float("inf"))
    effort_limits = torch.nan_to_num(effort_limits, nan=float("inf"), posinf=float("inf"))
    clipped = torch.clamp(computed, min=-effort_limits, max=effort_limits)
    return clipped, computed - clipped


def detach_terms(terms: dict[str, Any]) -> dict[str, Any]:
    """Detach tensors for storage/reporting without changing keys."""

    detached: dict[str, Any] = {}
    for key, value in terms.items():
        detached[key] = value.detach() if isinstance(value, torch.Tensor) else value
    return detached
