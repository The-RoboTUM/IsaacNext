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
from isaaclab.tendons.data_recording import DataRecording, DataRecordingConfig, motor_torque_tensor
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
    kinematic_consistency_threshold: float | None = 10.0
    kinematic_drop_before: int = 1
    kinematic_drop_after: int = 1
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
        self._previous_kinematic_joint_pos = self.robot.data.joint_pos.clone()
        self._previous_kinematic_joint_vel = self.robot.data.joint_vel.clone()
        self._previous_kinematic_time = 0.0
        self._record_env_ids = tuple(options.env_ids) if options.env_ids is not None else tuple(range(env.num_envs))
        self._retired_env_ids: set[int] = set()
        self._retired_env_reasons: dict[int, str] = {}
        self._skip_remaining_by_env: dict[int, int] = {}
        self._dropped_kinematic_rows = 0
        self._skipped_kinematic_rows = 0
        self._target_row_count = self._compute_target_row_count()

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
                "reset_sample_policy": "drop local frame window around done/reset samples",
                "kinematic_consistency_threshold_rad": options.kinematic_consistency_threshold,
                "kinematic_consistency_policy": "drop local frame window around inconsistent recorded-joint q/dq steps",
                "kinematic_drop_before": int(options.kinematic_drop_before),
                "kinematic_drop_after": int(options.kinematic_drop_after),
                "target_row_count": self._target_row_count,
                "target_row_count_policy": (
                    "record_max_steps defines target potential rows after stride/start filters; recording continues "
                    "past that step count until row_count reaches the target"
                ),
            },
        )
        self._record_joint_indices = sorted(
            {
                joint_index
                for side in self.recorder._selected_sides()
                for joint_index in self.recorder._joint_indices_by_side[side]
            }
        )

    @property
    def output_dir(self) -> Path:
        return self.recorder.output_dir

    def _compute_target_row_count(self) -> int | None:
        if self.options.max_steps is None:
            return None

        sampled_steps = 0
        stride = max(int(self.options.stride), 1)
        start_time = float(self.options.start_time)
        step_dt = float(self.env.step_dt)
        for step_index in range(1, int(self.options.max_steps) + 1):
            if step_index % stride != 0:
                continue
            if step_index * step_dt < start_time:
                continue
            sampled_steps += 1

        side_count = 2 if self.options.side == "both" else 1
        return sampled_steps * len(self._record_env_ids) * side_count

    def _target_row_count_reached(self) -> bool:
        return self._target_row_count is not None and self.recorder._row_count >= self._target_row_count

    def record_after_step(self, *, dones=None) -> bool:
        """Record the current post-step env state. Returns false once the row target is reached."""

        if self._closed:
            return False
        if self._target_row_count_reached():
            return False

        self.step_index += 1
        sim_time = self.step_index * float(self.env.step_dt)
        done_env_ids = {env_id for env_id in _done_env_ids(dones) if env_id in self._record_env_ids}
        inconsistent_env_ids = self._inconsistent_kinematic_env_ids(sim_time)
        new_unreliable_env_ids = done_env_ids | inconsistent_env_ids
        if new_unreliable_env_ids:
            self._dropped_kinematic_rows += self.recorder.drop_recent_samples(
                new_unreliable_env_ids,
                count=int(self.options.kinematic_drop_before),
            )
            for env_id in new_unreliable_env_ids:
                self._skip_remaining_by_env[env_id] = max(
                    self._skip_remaining_by_env.get(env_id, 0),
                    int(self.options.kinematic_drop_after),
                )

        skip_env_ids = {
            env_id
            for env_id in self._record_env_ids
            if self._skip_remaining_by_env.get(env_id, 0) > 0 or env_id in new_unreliable_env_ids
        }
        self._skipped_kinematic_rows += len(skip_env_ids)
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
            skip_env_ids=skip_env_ids,
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
                skip_env_ids=skip_env_ids,
            )
        if self._target_row_count is not None and self.recorder._row_count > self._target_row_count:
            trimmed = self.recorder.trim_to_row_count(self._target_row_count)
            self.recorder._context_metadata["target_row_count_trimmed_rows"] = int(
                self.recorder._context_metadata.get("target_row_count_trimmed_rows", 0)
            ) + int(trimmed)
        self._decrement_skip_windows(skip_env_ids - new_unreliable_env_ids)
        return not self._target_row_count_reached()

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
        self.recorder._context_metadata["dropped_kinematic_rows"] = int(self._dropped_kinematic_rows)
        self.recorder._context_metadata["skipped_kinematic_rows"] = int(self._skipped_kinematic_rows)
        self.recorder._context_metadata["actual_recording_steps"] = int(self.step_index)
        self.recorder._context_metadata["target_row_count"] = self._target_row_count
        self.recorder._context_metadata["target_row_count_reached"] = self._target_row_count_reached()
        self.recorder.close()
        self._closed = True

    def _tau_input_tensor(self):
        if self.params.recording.tau_source != "controller_plus_ground":
            return _tau_source_tensor(self.robot, self.params.recording.tau_source)

        tau = motor_torque_tensor(self.robot)
        if self.contact_sensor is not None:
            contact_force, contact_moment = _projected_contact_sensor_torque_parts(
                self.robot, self.contact_sensor, self.contact_body_names
            )
            contact_force_norm = torch.linalg.norm(contact_force, dim=-1, keepdim=True)
            contact_moment_norm = torch.linalg.norm(contact_moment, dim=-1, keepdim=True)
            contact_moment_valid = contact_moment_norm <= 2.0 * torch.clamp(contact_force_norm, min=1.0e-6)
            tau += contact_force + torch.where(contact_moment_valid, contact_moment, torch.zeros_like(contact_moment))
        return tau

    def _inconsistent_kinematic_env_ids(self, sim_time: float) -> set[int]:
        threshold = self.options.kinematic_consistency_threshold
        if threshold is None:
            self._update_previous_kinematic_state(sim_time)
            return set()

        dt = float(sim_time) - float(self._previous_kinematic_time)
        if dt <= 1.0e-9:
            self._update_previous_kinematic_state(sim_time)
            return set()

        joint_indices = self._record_joint_indices
        current_pos = self.robot.data.joint_pos[:, joint_indices]
        current_vel = self.robot.data.joint_vel[:, joint_indices]
        previous_pos = self._previous_kinematic_joint_pos[:, joint_indices]
        previous_vel = self._previous_kinematic_joint_vel[:, joint_indices]
        position_error = current_pos - previous_pos - 0.5 * (previous_vel + current_vel) * dt
        max_error = position_error.abs().amax(dim=1)
        bad_env_ids = {
            int(env_id)
            for env_id in torch.nonzero(max_error > float(threshold), as_tuple=False).flatten().cpu().tolist()
            if int(env_id) in self._record_env_ids and int(env_id) not in self._retired_env_ids
        }
        self._update_previous_kinematic_state(sim_time)
        return bad_env_ids

    def _update_previous_kinematic_state(self, sim_time: float) -> None:
        self._previous_kinematic_joint_pos = self.robot.data.joint_pos.clone()
        self._previous_kinematic_joint_vel = self.robot.data.joint_vel.clone()
        self._previous_kinematic_time = float(sim_time)

    def _decrement_skip_windows(self, skipped_env_ids: set[int]) -> None:
        for env_id in skipped_env_ids:
            remaining = self._skip_remaining_by_env.get(env_id, 0)
            if remaining <= 1:
                self._skip_remaining_by_env.pop(env_id, None)
            else:
                self._skip_remaining_by_env[env_id] = remaining - 1


def parse_env_ids(value: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated env ids from CLI."""

    if value is None or value.strip() == "":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _done_env_ids(dones) -> set[int]:
    if dones is None:
        return set()
    if isinstance(dones, torch.Tensor):
        done_tensor = dones.detach().to(dtype=torch.bool)
        if done_tensor.ndim == 0:
            return {0} if bool(done_tensor.item()) else set()
        if done_tensor.ndim > 1:
            done_tensor = done_tensor.reshape(done_tensor.shape[0], -1).any(dim=1)
        return {int(index) for index in torch.nonzero(done_tensor, as_tuple=False).flatten().cpu().tolist()}

    return {index for index, done in enumerate(dones) if bool(done)}


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
    if tau_source == "motor_torque":
        return motor_torque_tensor(robot)
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
    pantograph_spring = _pantograph_spring_torque(robot)
    pantograph_damping = _pantograph_damping_torque(robot)
    pantograph_applied_actuation = _pantograph_actuation_torque(robot)
    pantograph_actuation = pantograph_applied_actuation
    motor_actuation = motor_torque_tensor(robot)
    knee_flexor_actuation = _knee_flexor_actuation_torque(robot, motor_actuation)
    actuation_command = motor_actuation - knee_flexor_actuation
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
            "motor_actuation": motor_actuation,
            "knee_flexor_actuation": knee_flexor_actuation,
            "actuation_command": actuation_command,
            "contact_validated": contact_validated,
            "tendon": tendon,
            "pantograph_spring": pantograph_spring,
            "pantograph_damping": pantograph_damping,
            "pantograph_actuation": pantograph_actuation,
            "pantograph_applied_actuation": pantograph_applied_actuation,
        }

    solver_joint = robot.root_physx_view.get_dof_projected_joint_forces()
    physx_actuation = robot.root_physx_view.get_dof_actuation_forces()
    pantograph_computed_actuation = _pantograph_computed_actuation_torque(robot)
    pantograph_reconstructed_actuation = _pantograph_reconstructed_actuation_torque(
        robot,
        pantograph_spring=pantograph_spring,
        pantograph_damping=pantograph_damping,
    )
    pantograph_actuation_error = pantograph_applied_actuation - pantograph_reconstructed_actuation
    actuation = actuation_command
    joint_limit_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    joint_limit_upper = robot.data.soft_joint_pos_limits[:, :, 1]
    joint_limit_distance_lower = robot.data.joint_pos - joint_limit_lower
    joint_limit_distance_upper = joint_limit_upper - robot.data.joint_pos
    joint_limit_distance_min = torch.minimum(joint_limit_distance_lower, joint_limit_distance_upper)
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
    conservative = inertia + gravity + coriolis + tendon
    non_conservative = actuation_command + contact_validated + friction
    residual = conservative - non_conservative
    with_pantograph_actuation = actuation_command + pantograph_actuation
    with_knee_flexor_actuation = actuation_command + knee_flexor_actuation
    with_pantograph_and_knee_flexor_actuation = actuation_command + pantograph_actuation + knee_flexor_actuation
    residual_with_pantograph_actuation = conservative - (with_pantograph_actuation + contact_validated + friction)
    residual_with_knee_flexor_actuation = conservative - (with_knee_flexor_actuation + contact_validated + friction)
    residual_with_pantograph_and_knee_flexor_actuation = conservative - (
        with_pantograph_and_knee_flexor_actuation + contact_validated + friction
    )
    residual_no_pantograph_actuation = residual
    residual_no_knee_flexor_actuation = residual
    residual_no_pantograph_no_knee_flexor_actuation = residual
    residual_no_pantograph_no_knee_flexor_plus_solver = conservative - (
        actuation_command + contact_validated + friction + solver_constraint_internal
    )

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
        "motor_actuation": motor_actuation,
        "knee_flexor_actuation": knee_flexor_actuation,
        "pantograph_damping": pantograph_damping,
        "solver_joint": solver_joint,
        "actuation": actuation,
        "actuation_command": actuation_command,
        "pantograph_actuation": pantograph_actuation,
        "pantograph_applied_actuation": pantograph_applied_actuation,
        "pantograph_computed_actuation": pantograph_computed_actuation,
        "pantograph_reconstructed_actuation": pantograph_reconstructed_actuation,
        "pantograph_actuation_error": pantograph_actuation_error,
        "physx_actuation": physx_actuation,
        "solver_constraint_internal": solver_constraint_internal,
        "contact": contact,
        "contact_force": contact_force,
        "contact_moment": contact_moment,
        "contact_validated": contact_validated,
        **contact_groups,
        "tendon": tendon,
        "pantograph_spring": pantograph_spring,
        "tendon_model": tendon_model,
        "tendon_projection_delta": tendon_projection_delta,
        "residual": residual,
        "residual_with_pantograph_actuation": residual_with_pantograph_actuation,
        "residual_with_knee_flexor_actuation": residual_with_knee_flexor_actuation,
        "residual_with_pantograph_and_knee_flexor_actuation": residual_with_pantograph_and_knee_flexor_actuation,
        "residual_no_pantograph_actuation": residual_no_pantograph_actuation,
        "residual_no_knee_flexor_actuation": residual_no_knee_flexor_actuation,
        "residual_no_pantograph_no_knee_flexor_actuation": residual_no_pantograph_no_knee_flexor_actuation,
        "residual_no_pantograph_no_knee_flexor_plus_solver": residual_no_pantograph_no_knee_flexor_plus_solver,
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


def _pantograph_spring_torque(robot):
    spring = torch.zeros_like(robot.data.joint_pos)
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return spring

    stiffness = robot.data.joint_stiffness[:, pantograph_indices]
    target = robot.data.joint_pos_target[:, pantograph_indices]
    position = robot.data.joint_pos[:, pantograph_indices]
    spring[:, pantograph_indices] = stiffness * (position - target)
    return spring


def _pantograph_damping_torque(robot):
    damping_torque = torch.zeros_like(robot.data.joint_pos)
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return damping_torque

    damping = robot.data.joint_damping[:, pantograph_indices]
    target_velocity = robot.data.joint_vel_target[:, pantograph_indices]
    velocity = robot.data.joint_vel[:, pantograph_indices]
    damping_torque[:, pantograph_indices] = damping * (target_velocity - velocity)
    return damping_torque


def _pantograph_actuation_torque(robot):
    actuation = torch.zeros_like(robot.data.joint_pos)
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return actuation

    actuation[:, pantograph_indices] = robot.data.applied_torque[:, pantograph_indices]
    return actuation


def _pantograph_computed_actuation_torque(robot):
    actuation = torch.zeros_like(robot.data.joint_pos)
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return actuation

    actuation[:, pantograph_indices] = robot.data.computed_torque[:, pantograph_indices]
    return actuation


def _pantograph_reconstructed_actuation_torque(
    robot,
    *,
    pantograph_spring: torch.Tensor,
    pantograph_damping: torch.Tensor,
):
    actuation = -pantograph_spring + pantograph_damping
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return torch.zeros_like(robot.data.joint_pos)

    effort_target = torch.zeros_like(robot.data.joint_pos)
    effort_target[:, pantograph_indices] = robot.data.joint_effort_target[:, pantograph_indices]
    return actuation + effort_target


def _knee_flexor_actuation_torque(robot, actuation: torch.Tensor):
    knee_flexor = torch.zeros_like(actuation)
    knee_flexor_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("l8_knee_flexor", "r8_knee_flexor")
    ]
    if not knee_flexor_indices:
        return knee_flexor

    knee_flexor[:, knee_flexor_indices] = actuation[:, knee_flexor_indices]
    return knee_flexor


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

    # The cached link wrenches are opposite the generalized-force sign used by
    # cached_tendon_joint_torques and the database force-balance convention.
    return -tau_tendon


def _actual_generalized_force(
    robot,
    joint_ids: list[int],
    *,
    force_api_name: str,
    compensation_api_name: str,
):
    try:
        return _compensation_generalized_force(robot, joint_ids, compensation_api_name=compensation_api_name)
    except Exception:
        return getattr(robot.root_physx_view, force_api_name)()[:, joint_ids]


def _compensation_generalized_force(robot, joint_ids: list[int], *, compensation_api_name: str):
    compensation = getattr(robot.root_physx_view, compensation_api_name)()
    generalized_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    return compensation[:, generalized_joint_ids]
