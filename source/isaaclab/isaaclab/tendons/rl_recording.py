# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in Forrest database recording helpers for RL play/evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from isaaclab.sensors import ContactSensor
from isaaclab.tendons.data_recording import DataRecording, DataRecordingConfig
from isaaclab.tendons.parameter_loader import ForrestParameterConfig
from isaaclab.utils.math import quat_apply

CONTACT_GROUP_PATTERNS = {
    "digit": ("digit",),
    "connector": ("foot_connector",),
    "base": ("base", "hip", "differential_cage"),
}


@dataclass(frozen=True)
class ForrestRLRecordingOptions:
    """Runtime options for opt-in RL recording."""

    enabled: bool = False
    output_dir: str | None = None
    side: str = "left"
    env_ids: tuple[int, ...] | None = None
    overwrite: bool = False
    stride: int = 1
    start_time: float = 0.0
    record_dynamics: bool = True
    record_debug_dynamics: bool = False
    residual_filter_threshold: float | None = None
    max_steps: int | None = None
    contact_sensor_name: str = "contact_forces"
    robot_name: str = "robot"


class ForrestRLRecorder:
    """Adapter that records Forrest RL env state through the shared database writer."""

    def __init__(
        self,
        env,
        *,
        params: ForrestParameterConfig,
        options: ForrestRLRecordingOptions,
        task_name: str | None = None,
        checkpoint: str | None = None,
    ):
        self.env = env
        self.params = params
        self.options = options
        self.step_index = 0
        self._closed = False

        self.robot = env.scene.articulations[options.robot_name]
        self.contact_sensor: ContactSensor | None = env.scene.sensors.get(options.contact_sensor_name)
        self.contact_body_names = tuple(params.training.contacts.contact_sensor_body_names)
        self.tendon_manager = _find_tendon_manager(env)
        self.actuated_joint_indices = _actuated_joint_indices(self.robot)
        self._previous_record_joint_vel = self.robot.data.joint_vel.clone()
        self._previous_record_root_vel = self.robot.data.root_com_vel_w.clone()
        self._previous_record_time = 0.0

        output_dir = options.output_dir or _timestamped_output_dir(params.run.output_dir)
        self.recorder = DataRecording(
            DataRecordingConfig(
                output_dir=output_dir,
                sqlite_filename=params.recording.kinematics_db_filename,
                tendon_sqlite_filename=params.recording.tendons_db_filename,
                dynamics_sqlite_filename=params.recording.dynamics_db_filename,
                debug_sqlite_filename=params.recording.debug_dynamics_db_filename,
                metadata_filename=params.recording.metadata_filename,
                viz_vars_filename=params.recording.viz_vars_filename,
                joint_set=params.recording.joint_set,
                side_policy=_side_policy(options.side),
                selected_env_ids=options.env_ids,
                body_set=params.recording.body_set,
                record_spatial_state=False,
                sampling_stride=options.stride,
                startup_skip_seconds=options.start_time,
                constraint_mode="rl_play",
                controller="policy",
                tau_source=params.recording.tau_source,
                record_tendons=False,
                record_dynamics=options.record_dynamics,
                record_debug_dynamics=options.record_debug_dynamics,
                residual_filter_threshold=options.residual_filter_threshold,
                overwrite=options.overwrite,
                parameter_file=None,
            )
        )
        self.recorder.initialize(
            self.robot,
            sim_dt=env.step_dt,
            metadata={
                "mode": "rl_play",
                "task": task_name,
                "checkpoint": checkpoint,
                "num_envs": int(env.num_envs),
                "recorded_env_ids": list(options.env_ids) if options.env_ids is not None else "all",
                "contact_sensor_name": options.contact_sensor_name,
                "contact_body_names": list(self.contact_body_names),
                "physics_dt": float(env.physics_dt),
                "step_dt": float(env.step_dt),
                "decimation": int(env.cfg.decimation),
            },
        )

    @property
    def output_dir(self) -> Path:
        return self.recorder.output_dir

    def record_after_step(self) -> bool:
        """Record the current post-step env state. Returns false once max_steps is reached."""

        if self._closed:
            return False
        if self.options.max_steps is not None and self.step_index >= self.options.max_steps:
            return False

        self.step_index += 1
        sim_time = self.step_index * float(self.env.step_dt)
        tau_input = self._tau_input_tensor()
        joint_acc_recording, root_acc_recording = self._recording_accelerations(sim_time)
        dynamics_terms = None
        if (
            self.recorder.cfg.record_dynamics
            or self.recorder.cfg.record_debug_dynamics
            or self.recorder.cfg.residual_filter_threshold is not None
        ):
            dynamics_terms = _dynamics_terms(
                self.robot,
                contact_sensor=self.contact_sensor,
                contact_body_names=self.contact_body_names,
                tendon_manager=self.tendon_manager,
                joint_acc_for_inertia=joint_acc_recording,
                root_acc_for_inertia=root_acc_recording,
                include_debug=(
                    self.recorder.cfg.record_debug_dynamics or self.recorder.cfg.residual_filter_threshold is not None
                ),
            )
        self.recorder.record_step(
            step_index=self.step_index,
            sim_time=sim_time,
            robot=self.robot,
            extra_context={"source": "rsl_rl_play"},
            tau_override=tau_input,
            ddq_override=joint_acc_recording,
            dynamics_terms=dynamics_terms,
        )
        if self.recorder.cfg.record_dynamics or self.recorder.cfg.record_debug_dynamics:
            if dynamics_terms is None:
                raise RuntimeError("Internal error: dynamics terms were not computed for dynamics recording.")
            self.recorder.record_dynamics_step(
                step_index=self.step_index,
                sim_time=sim_time,
                robot=self.robot,
                dynamics_terms=dynamics_terms,
                tau_input=tau_input,
            )
        return self.options.max_steps is None or self.step_index < self.options.max_steps

    def _recording_accelerations(self, sim_time: float):
        joint_vel = self.robot.data.joint_vel
        root_vel = self.robot.data.root_com_vel_w
        dt = max(float(sim_time) - float(self._previous_record_time), 1.0e-9)
        joint_acc = (joint_vel - self._previous_record_joint_vel) / dt
        root_acc = (root_vel - self._previous_record_root_vel) / dt
        self._previous_record_joint_vel = joint_vel.clone()
        self._previous_record_root_vel = root_vel.clone()
        self._previous_record_time = float(sim_time)
        return joint_acc, root_acc

    def close(self) -> None:
        if self._closed:
            return
        self.recorder.close()
        self._closed = True

    def _tau_input_tensor(self):
        if self.params.recording.tau_source != "controller_plus_ground":
            return _tau_source_tensor(self.robot, self.params.recording.tau_source)

        tau = torch.zeros_like(self.robot.data.joint_pos)
        tau[:, self.actuated_joint_indices] = self.robot.data.applied_torque[:, self.actuated_joint_indices]
        if self.contact_sensor is not None:
            contact_force, contact_moment = _projected_contact_sensor_torque_parts(
                self.robot, self.contact_sensor, self.contact_body_names
            )
            contact_force_norm = torch.linalg.norm(contact_force, dim=-1, keepdim=True)
            contact_moment_norm = torch.linalg.norm(contact_moment, dim=-1, keepdim=True)
            contact_moment_valid = contact_moment_norm <= 2.0 * torch.clamp(contact_force_norm, min=1.0e-6)
            tau += contact_force + torch.where(contact_moment_valid, contact_moment, torch.zeros_like(contact_moment))
        return tau


def parse_env_ids(value: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated env ids from CLI."""

    if value is None or value.strip() == "":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _timestamped_output_dir(root: str | Path) -> str:
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(root) / f"forrest_dbs_{datestamp}")


def _side_policy(side: str) -> str:
    if side == "left":
        return "left_only"
    if side == "right":
        return "right_only"
    if side == "both":
        return "both_as_samples"
    raise ValueError(f"Unsupported recording side: {side}")


def _actuated_joint_indices(robot) -> list[int]:
    nonzero = torch.any(robot.data.default_joint_stiffness != 0.0, dim=0)
    return [int(index) for index in torch.nonzero(nonzero, as_tuple=False).squeeze(-1).tolist()]


def _tau_source_tensor(robot, tau_source: str):
    if tau_source == "applied_torque":
        return robot.data.applied_torque
    if tau_source == "computed_torque":
        return robot.data.computed_torque
    if tau_source == "zero":
        return robot.data.joint_pos * 0.0
    raise ValueError(f"Unsupported RL recording tau source: {tau_source}")


def _projected_contact_sensor_torque_parts(robot, contact_sensor: ContactSensor, contact_body_names: tuple[str, ...]):
    tau_force = torch.zeros_like(robot.data.joint_pos)
    tau_moment = torch.zeros_like(robot.data.joint_pos)
    available_names = set(contact_sensor.body_names)
    selected_names = tuple(name for name in contact_body_names if name in available_names)
    if not selected_names:
        return tau_force, tau_moment

    sensor_body_indices = [contact_sensor.body_names.index(name) for name in selected_names]
    robot_body_indices, _ = robot.find_bodies(list(selected_names), preserve_order=True)

    if contact_sensor.data.force_matrix_w is not None:
        normal_forces_by_filter = contact_sensor.data.force_matrix_w[:, sensor_body_indices, :, :]
    else:
        normal_forces_by_filter = contact_sensor.data.net_forces_w[:, sensor_body_indices, :].unsqueeze(2)
    friction_forces = getattr(contact_sensor.data, "friction_forces_w", None)
    if friction_forces is not None:
        friction_forces_by_filter = friction_forces[:, sensor_body_indices, :, :]
    else:
        friction_forces_by_filter = torch.zeros_like(normal_forces_by_filter)
    forces_by_filter = normal_forces_by_filter + friction_forces_by_filter
    forces_world = forces_by_filter.sum(dim=2)
    if not torch.any(forces_world).item():
        return tau_force, tau_moment

    contact_pos = getattr(contact_sensor.data, "contact_pos_w", None)
    if contact_pos is not None:
        selected_contact_pos = contact_pos[:, sensor_body_indices, :, :]
        valid_contact = torch.isfinite(selected_contact_pos).all(dim=-1)
        force_weights = torch.linalg.norm(forces_by_filter, dim=-1) * valid_contact.to(dtype=forces_by_filter.dtype)
        weight_sum = force_weights.sum(dim=2, keepdim=True).clamp_min(1.0e-12)
        contact_pos_world = (torch.nan_to_num(selected_contact_pos, nan=0.0) * force_weights.unsqueeze(-1)).sum(
            dim=2
        ) / weight_sum
    else:
        contact_pos_world = None

    num_joints = robot.data.joint_pos.shape[1]
    joint_ids = list(range(num_joints))
    jacobian_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    jacobians = robot.root_physx_view.get_jacobians()

    for local_foot_index, body_index in enumerate(robot_body_indices):
        jacobian_body_index = int(body_index) - 1 if robot.is_fixed_base else int(body_index)
        jacobian_linear = jacobians[:, jacobian_body_index, 0:3, :][:, :, jacobian_joint_ids]
        jacobian_angular = jacobians[:, jacobian_body_index, 3:6, :][:, :, jacobian_joint_ids]
        force = forces_world[:, local_foot_index, :]
        body_pos = robot.data.body_pos_w[:, int(body_index), :]
        if contact_pos_world is not None:
            moment = torch.cross(contact_pos_world[:, local_foot_index, :] - body_pos, force, dim=1)
        else:
            moment = torch.zeros_like(force)
        tau_force += torch.bmm(jacobian_linear.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
        tau_moment += torch.bmm(jacobian_angular.transpose(1, 2), moment.unsqueeze(-1)).squeeze(-1)
    return tau_force, tau_moment


def _projected_contact_sensor_torque_groups(
    robot,
    contact_sensor: ContactSensor,
    contact_body_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    grouped: dict[str, torch.Tensor] = {}
    for group_name, patterns in CONTACT_GROUP_PATTERNS.items():
        names = tuple(name for name in contact_body_names if any(pattern in name for pattern in patterns))
        tau_force, tau_moment = _projected_contact_sensor_torque_parts(robot, contact_sensor, names)
        grouped[f"contact_{group_name}_force"] = tau_force
        grouped[f"contact_{group_name}_moment"] = tau_moment
        grouped[f"contact_{group_name}"] = tau_force + tau_moment
    return grouped


def _zero_contact_sensor_torque_groups(robot) -> dict[str, torch.Tensor]:
    grouped: dict[str, torch.Tensor] = {}
    for group_name in CONTACT_GROUP_PATTERNS:
        zero = torch.zeros_like(robot.data.joint_pos)
        grouped[f"contact_{group_name}_force"] = zero
        grouped[f"contact_{group_name}_moment"] = zero
        grouped[f"contact_{group_name}"] = zero
    return grouped


def _projected_contact_sensor_torque(robot, contact_sensor: ContactSensor, contact_body_names: tuple[str, ...]):
    tau_force, tau_moment = _projected_contact_sensor_torque_parts(robot, contact_sensor, contact_body_names)
    return tau_force + tau_moment


def _dynamics_terms(
    robot,
    *,
    contact_sensor: ContactSensor | None = None,
    contact_body_names: tuple[str, ...] = (),
    tendon_manager=None,
    joint_acc_for_inertia=None,
    root_acc_for_inertia=None,
    include_debug: bool = False,
) -> dict[str, Any]:
    num_joints = robot.data.joint_pos.shape[1]
    joint_ids = list(range(num_joints))

    mass_matrices_full = robot.root_physx_view.get_generalized_mass_matrices()
    mass_matrices = mass_matrices_full if robot.is_fixed_base else mass_matrices_full[:, 6:, 6:]
    raw_joint_acc = robot.data.joint_acc
    joint_acc = raw_joint_acc if joint_acc_for_inertia is None else joint_acc_for_inertia
    inertia_joint_all = torch.bmm(mass_matrices, joint_acc.unsqueeze(-1)).squeeze(-1)
    inertia_joint_only = inertia_joint_all
    inertia_leg_self = torch.zeros_like(inertia_joint_all)
    inertia_other_joints = torch.zeros_like(inertia_joint_all)
    inertia_raw = torch.bmm(mass_matrices, raw_joint_acc.unsqueeze(-1)).squeeze(-1)
    if robot.is_fixed_base:
        inertia_root_coupling = torch.zeros_like(inertia_joint_all)
        inertia_root_coupling_raw = torch.zeros_like(inertia_joint_all)
        inertia_root_coupling_alt = torch.zeros_like(inertia_joint_all)
        inertia_root_coupled_alt = inertia_joint_all
        inertia_full_raw = inertia_raw
        inertia_recording_interval = inertia_joint_all
    else:
        link_accelerations = robot.root_physx_view.get_link_accelerations()
        root_acc_raw = link_accelerations[:, 0, :]
        root_acc = root_acc_raw if root_acc_for_inertia is None else root_acc_for_inertia
        root_acc_alt = torch.cat((root_acc_raw[:, 3:6], root_acc_raw[:, 0:3]), dim=1)
        inertia_root_coupling = torch.bmm(mass_matrices_full[:, 6:, :6], root_acc.unsqueeze(-1)).squeeze(-1)
        inertia_root_coupling_raw = torch.bmm(mass_matrices_full[:, 6:, :6], root_acc_raw.unsqueeze(-1)).squeeze(-1)
        inertia_root_coupling_alt = torch.bmm(mass_matrices_full[:, 6:, :6], root_acc_alt.unsqueeze(-1)).squeeze(-1)
        inertia_recording_interval = inertia_root_coupling + inertia_joint_all
        inertia_root_coupled_alt = inertia_root_coupling_alt + inertia_joint_all
        inertia_full_raw = inertia_root_coupling_raw + inertia_raw
    inertia = inertia_full_raw
    coriolis = _actual_generalized_force(
        robot,
        joint_ids,
        force_api_name="get_coriolis_and_centrifugal_forces",
        compensation_api_name="get_coriolis_and_centrifugal_compensation_forces",
    )
    gravity = _actual_generalized_force(
        robot,
        joint_ids,
        force_api_name="get_generalized_gravity_forces",
        compensation_api_name="get_gravity_compensation_forces",
    )

    dynamic = robot.data.joint_dynamic_friction_coeff
    viscous = robot.data.joint_viscous_friction_coeff
    friction_dynamic = -dynamic * torch.sign(robot.data.joint_vel)
    friction_viscous = -viscous * robot.data.joint_vel
    friction = friction_dynamic + friction_viscous
    actuation_command = robot.data.applied_torque
    if contact_sensor is not None:
        contact_force, contact_moment = _projected_contact_sensor_torque_parts(
            robot, contact_sensor, contact_body_names
        )
    else:
        contact_force = torch.zeros_like(robot.data.joint_pos)
        contact_moment = torch.zeros_like(robot.data.joint_pos)
    contact = contact_force + contact_moment
    contact_force_norm = torch.linalg.norm(contact_force, dim=-1, keepdim=True)
    contact_moment_norm = torch.linalg.norm(contact_moment, dim=-1, keepdim=True)
    contact_moment_valid = contact_moment_norm <= 2.0 * torch.clamp(contact_force_norm, min=1.0e-6)
    contact_validated = contact_force + torch.where(
        contact_moment_valid, contact_moment, torch.zeros_like(contact_moment)
    )
    tendon = _projected_tendon_wrench_torque(robot, tendon_manager)
    tendon_model = _tendon_joint_torque_tensor(robot, tendon_manager)
    tendon_projection_delta = tendon - tendon_model
    if not include_debug:
        return {
            "inertia": inertia,
            "coriolis": coriolis,
            "gravity": gravity,
            "friction": friction,
            "actuation_command": actuation_command,
            "contact_validated": contact_validated,
            "tendon": tendon,
        }

    solver_joint = robot.root_physx_view.get_dof_projected_joint_forces()
    physx_actuation = robot.root_physx_view.get_dof_actuation_forces()
    actuation = physx_actuation
    actuation_estimated = actuation_command
    actuation_estimated_hip = _estimated_hip_actuation(robot, actuation_command)
    actuation_estimated_hip_lateral_flexion = _estimated_hip_lateral_flexion_actuation(robot, actuation_command)
    actuation_estimated_passive = _estimated_passive_actuation(robot, actuation_command)
    joint_drive_pos_target = robot.data.joint_pos_target
    joint_drive_vel_target = robot.data.joint_vel_target
    joint_drive_effort_target = robot.data.joint_effort_target
    joint_drive_stiffness = robot.data.joint_stiffness
    joint_drive_damping = robot.data.joint_damping
    joint_effort_limit = robot.data.joint_effort_limits
    joint_velocity_limit = robot.data.soft_joint_vel_limits
    joint_limit_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    joint_limit_upper = robot.data.soft_joint_pos_limits[:, :, 1]
    joint_limit_distance_lower = robot.data.joint_pos - joint_limit_lower
    joint_limit_distance_upper = joint_limit_upper - robot.data.joint_pos
    joint_limit_distance_min = torch.minimum(joint_limit_distance_lower, joint_limit_distance_upper)
    drive_stiffness = joint_drive_stiffness * (joint_drive_pos_target - robot.data.joint_pos)
    drive_damping = joint_drive_damping * (joint_drive_vel_target - robot.data.joint_vel)
    drive_effort_target = joint_drive_effort_target
    drive_pd = drive_stiffness + drive_damping + drive_effort_target
    drive_pd_clipped = torch.clamp(drive_pd, -joint_effort_limit, joint_effort_limit)
    armature_inertia = robot.data.joint_armature * raw_joint_acc
    solver_constraint_passive = _passive_solver_constraint(robot, solver_joint)
    solver_constraint_limit = torch.where(
        joint_limit_distance_min <= 0.05, solver_joint, torch.zeros_like(solver_joint)
    )
    solver_constraint_internal = torch.where(
        (solver_constraint_passive != 0.0) | (solver_constraint_limit != 0.0),
        solver_joint,
        torch.zeros_like(solver_joint),
    )
    if contact_sensor is not None:
        contact_groups = _projected_contact_sensor_torque_groups(robot, contact_sensor, contact_body_names)
    else:
        contact_groups = _zero_contact_sensor_torque_groups(robot)
    required_forces = inertia - gravity - coriolis - tendon
    required_forces_recording_interval = inertia_recording_interval - gravity - coriolis - tendon
    applied_full_contact = actuation + contact + friction
    applied_force_only = actuation + contact_force + friction
    applied_validated = actuation + contact_validated + friction
    applied_estimated_actuation = actuation_estimated + contact_validated + friction
    applied_estimated_hip_actuation = actuation_estimated_hip + contact_validated + friction
    applied_estimated_hip_force_contact = actuation_estimated_hip + contact_force + friction
    applied_estimated_hip_force_contact_solver = (
        actuation_estimated_hip + contact_force + friction + solver_constraint_passive
    )
    applied_estimated_hip_lateral_flexion_force_contact_solver_internal = (
        actuation_estimated_hip_lateral_flexion + contact_force + friction + solver_constraint_internal
    )
    unmodeled_quasistatic = -gravity - coriolis - tendon - applied_validated
    unmodeled_full_contact = required_forces - applied_full_contact
    unmodeled_contact_force_only = required_forces - applied_force_only
    unmodeled_contact_validated = required_forces - applied_validated
    unmodeled_estimated_actuation = required_forces - applied_estimated_actuation
    unmodeled_estimated_hip_actuation = required_forces - applied_estimated_hip_actuation
    unmodeled_estimated_hip_force_contact = required_forces - applied_estimated_hip_force_contact
    unmodeled_estimated_hip_force_contact_solver = required_forces - applied_estimated_hip_force_contact_solver
    unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal = (
        required_forces - applied_estimated_hip_lateral_flexion_force_contact_solver_internal
    )
    unmodeled_full_dynamics = unmodeled_contact_validated
    unmodeled_recording_interval = required_forces_recording_interval - applied_validated
    unmodeled = unmodeled_contact_validated
    inverse_residual = unmodeled
    solver_residual = solver_joint - applied_validated

    return {
        "inertia": inertia,
        "inertia_recording_interval": inertia_recording_interval,
        "inertia_raw": inertia_raw,
        "inertia_joint_only": inertia_joint_only,
        "inertia_joint_all": inertia_joint_all,
        "inertia_leg_self": inertia_leg_self,
        "inertia_other_joints": inertia_other_joints,
        "inertia_root_coupling": inertia_root_coupling,
        "inertia_root_coupling_raw": inertia_root_coupling_raw,
        "inertia_root_coupling_alt": inertia_root_coupling_alt,
        "inertia_root_coupled_alt": inertia_root_coupled_alt,
        "inertia_full_raw": inertia_full_raw,
        "coriolis": coriolis,
        "gravity": gravity,
        "friction_dynamic": friction_dynamic,
        "friction_viscous": friction_viscous,
        "friction": friction,
        "solver_joint": solver_joint,
        "actuation": actuation,
        "actuation_command": actuation_command,
        "actuation_estimated": actuation_estimated,
        "actuation_estimated_hip": actuation_estimated_hip,
        "actuation_estimated_hip_lateral_flexion": actuation_estimated_hip_lateral_flexion,
        "actuation_estimated_passive": actuation_estimated_passive,
        "physx_actuation": physx_actuation,
        "solver_constraint_passive": solver_constraint_passive,
        "solver_constraint_limit": solver_constraint_limit,
        "solver_constraint_internal": solver_constraint_internal,
        "joint_drive_pos_target": joint_drive_pos_target,
        "joint_drive_vel_target": joint_drive_vel_target,
        "joint_drive_effort_target": joint_drive_effort_target,
        "joint_drive_stiffness": joint_drive_stiffness,
        "joint_drive_damping": joint_drive_damping,
        "joint_effort_limit": joint_effort_limit,
        "joint_velocity_limit": joint_velocity_limit,
        "joint_limit_lower": joint_limit_lower,
        "joint_limit_upper": joint_limit_upper,
        "joint_limit_distance_lower": joint_limit_distance_lower,
        "joint_limit_distance_upper": joint_limit_distance_upper,
        "joint_limit_distance_min": joint_limit_distance_min,
        "drive_stiffness": drive_stiffness,
        "drive_damping": drive_damping,
        "drive_effort_target": drive_effort_target,
        "drive_pd": drive_pd,
        "drive_pd_clipped": drive_pd_clipped,
        "armature_inertia": armature_inertia,
        "contact": contact,
        "contact_force": contact_force,
        "contact_moment": contact_moment,
        "contact_validated": contact_validated,
        **contact_groups,
        "tendon": tendon,
        "tendon_model": tendon_model,
        "tendon_projection_delta": tendon_projection_delta,
        "unmodeled_quasistatic": unmodeled_quasistatic,
        "unmodeled_full_dynamics": unmodeled_full_dynamics,
        "unmodeled_recording_interval": unmodeled_recording_interval,
        "unmodeled_estimated_actuation": unmodeled_estimated_actuation,
        "unmodeled_estimated_hip_actuation": unmodeled_estimated_hip_actuation,
        "unmodeled_estimated_hip_force_contact": unmodeled_estimated_hip_force_contact,
        "unmodeled_estimated_hip_force_contact_solver": unmodeled_estimated_hip_force_contact_solver,
        "unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal": (
            unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal
        ),
        "unmodeled_full_contact": unmodeled_full_contact,
        "unmodeled_contact_force_only": unmodeled_contact_force_only,
        "unmodeled_contact_validated": unmodeled_contact_validated,
        "unmodeled": unmodeled,
        "inverse_residual": inverse_residual,
        "solver_residual": solver_residual,
        "mass_matrix": mass_matrices,
        "joint_acc_for_inertia": joint_acc,
    }


def _estimated_hip_actuation(robot, actuation_command: torch.Tensor) -> torch.Tensor:
    estimated = torch.zeros_like(actuation_command)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if (
            "_acetabulofemoral_roll" in joint_name
            or "_acetabulofemoral_lateral" in joint_name
            or "_pseudo_acetabulofemoral_flexion" in joint_name
        ):
            estimated[:, joint_index] = actuation_command[:, joint_index]
    return estimated


def _estimated_hip_lateral_flexion_actuation(robot, actuation_command: torch.Tensor) -> torch.Tensor:
    estimated = torch.zeros_like(actuation_command)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if "_acetabulofemoral_lateral" in joint_name or "_pseudo_acetabulofemoral_flexion" in joint_name:
            estimated[:, joint_index] = actuation_command[:, joint_index]
    return estimated


def _estimated_passive_actuation(robot, actuation_command: torch.Tensor) -> torch.Tensor:
    estimated = torch.zeros_like(actuation_command)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if _is_passive_tendon_chain_joint(joint_name):
            estimated[:, joint_index] = actuation_command[:, joint_index]
    return estimated


def _passive_solver_constraint(robot, solver_joint: torch.Tensor) -> torch.Tensor:
    constraint = torch.zeros_like(solver_joint)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if _is_solver_constraint_diagnostic_joint(joint_name):
            constraint[:, joint_index] = solver_joint[:, joint_index]
    return constraint


def _is_passive_tendon_chain_joint(joint_name: str) -> bool:
    return any(
        token in joint_name
        for token in (
            "3b_femorotibial_back",
            "4b_intertarsal_back",
            "3f_femorotibial_front",
            "4f_intertarsal_front",
            "4p_intertarsal_pulley",
            "5_metatarsophalangeal",
            "6_interphalangeal",
        )
    )


def _is_solver_constraint_diagnostic_joint(joint_name: str) -> bool:
    return any(
        token in joint_name
        for token in (
            "3b_femorotibial_back",
            "4f_intertarsal_front",
            "4b_intertarsal_back",
            "4p_intertarsal_pulley",
            "5_metatarsophalangeal",
            "8_knee_flexor",
        )
    )


def _find_tendon_manager(env):
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None
    for term_name in getattr(action_manager, "active_terms", ()):
        try:
            term = action_manager.get_term(term_name)
        except Exception:
            continue
        tendon_manager = getattr(term, "tendon_manager", None)
        if tendon_manager is not None:
            return tendon_manager
    return None


def _tendon_joint_torque_tensor(robot, tendon_manager):
    if tendon_manager is None:
        return torch.zeros_like(robot.data.joint_pos)
    tendon = getattr(tendon_manager, "cached_tendon_joint_torques", None)
    if tendon is None:
        return torch.zeros_like(robot.data.joint_pos)
    return tendon


def _projected_tendon_wrench_torque(robot, tendon_manager):
    if tendon_manager is None:
        return torch.zeros_like(robot.data.joint_pos)

    link_torques = getattr(tendon_manager, "cached_tendon_link_torques", None)
    link_forces = getattr(tendon_manager, "cached_tendon_forces", None)
    body_ids = getattr(tendon_manager, "cached_tendon_body_ids", None)
    if link_torques is None or body_ids is None:
        return _tendon_joint_torque_tensor(robot, tendon_manager)

    body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=robot.device).flatten()
    if body_ids.numel() == 0:
        return torch.zeros_like(robot.data.joint_pos)

    if link_forces is None:
        link_forces = torch.zeros_like(link_torques)

    num_joints = robot.data.joint_pos.shape[1]
    joint_ids = list(range(num_joints))
    jacobian_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    jacobians = robot.root_physx_view.get_jacobians()
    tau_tendon = torch.zeros_like(robot.data.joint_pos)

    for local_body_index, body_id in enumerate(body_ids.tolist()):
        body_index = int(body_id)
        jacobian_body_index = body_index - 1 if robot.is_fixed_base else body_index
        jacobian_linear = jacobians[:, jacobian_body_index, 0:3, :][:, :, jacobian_joint_ids]
        jacobian_angular = jacobians[:, jacobian_body_index, 3:6, :][:, :, jacobian_joint_ids]
        body_quat = robot.data.body_quat_w[:, body_index, :]
        force_world = quat_apply(body_quat, link_forces[:, local_body_index, :])
        torque_world = quat_apply(body_quat, link_torques[:, local_body_index, :])
        tau_tendon += torch.bmm(jacobian_linear.transpose(1, 2), force_world.unsqueeze(-1)).squeeze(-1)
        tau_tendon += torch.bmm(jacobian_angular.transpose(1, 2), torque_world.unsqueeze(-1)).squeeze(-1)

    return tau_tendon


def _actual_generalized_force(robot, joint_ids: list[int], *, force_api_name: str, compensation_api_name: str):
    try:
        compensation = getattr(robot.root_physx_view, compensation_api_name)()
        generalized_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
        return -compensation[:, generalized_joint_ids]
    except Exception:
        pass

    try:
        return getattr(robot.root_physx_view, force_api_name)()[:, joint_ids]
    except Exception:
        raise
