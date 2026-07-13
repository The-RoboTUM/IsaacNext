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
from isaaclab.tendons.data_recording import (
    BASE_COORDINATE_COUNT,
    DataRecording,
    DataRecordingConfig,
    actuation_command_tensor,
    motor_torque_tensor,
)
from isaaclab.tendons.parameter_loader import ForrestParameterConfig
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_box_plus, quat_error_magnitude

CONTACT_GROUP_PATTERNS = {
    "digit": ("digit",),
    "connector": ("foot_connector",),
    "base": ("base", "hip", "differential_cage"),
    "self_collision": ("s23",),
}


@dataclass(frozen=True)
class ForrestRLRecordingOptions:
    """Runtime options for opt-in RL recording."""

    enabled: bool = False
    output_dir: str | None = None
    side: str = "full"
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
        self.contact_detail_sensors = _find_contact_detail_sensors(env, options.contact_sensor_name)
        self.contact_body_names = _recording_contact_body_names(
            params,
            primary_sensor=self.contact_sensor,
            detail_sensors=self.contact_detail_sensors,
        )
        self.tendon_manager = _find_tendon_manager(env)
        self.actuated_joint_indices = _actuated_joint_indices(self.robot)
        self._previous_record_joint_vel = self.robot.data.joint_vel.clone()
        self._previous_record_root_vel = self.robot.data.root_com_vel_w.clone()
        self._previous_record_time = 0.0
        self._previous_kinematic_joint_pos = self.robot.data.joint_pos.clone()
        self._previous_kinematic_joint_vel = self.robot.data.joint_vel.clone()
        self._previous_kinematic_root_pos = self.robot.data.root_pos_w.clone()
        self._previous_kinematic_root_quat = self.robot.data.root_quat_w.clone()
        self._previous_kinematic_root_vel = self.robot.data.root_com_vel_w.clone()
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
                record_base_state=bool(params.recording.record_base_state or options.side == "full"),
                record_spatial_state=False,
                sampling_stride=options.stride,
                startup_skip_seconds=options.start_time,
                constraint_mode="rl_play",
                controller="policy",
                tau_source=params.recording.tau_source,
                ddq_source=params.recording.ddq_source,
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
                "contact_detail_sensor_names": [name for name, _sensor in self.contact_detail_sensors],
                "contact_body_names": list(self.contact_body_names),
                "contact_measurement_policy": (
                    "prefer one-body recording detail sensors for contact points/friction; fall back to the main "
                    "multi-body contact sensor when detail sensors are unavailable"
                ),
                "physics_dt": float(env.physics_dt),
                "step_dt": float(env.step_dt),
                "decimation": int(env.cfg.decimation),
                "reset_sample_policy": "drop local frame window around done/reset samples",
                "model_parameter_randomization_policy": (
                    "rsl_rl/play.py disables base_com, add_base_mass, base_external_force_torque, and push_robot "
                    "events when --record_forrest_dbs is active; actual masses/COMs are still exported in metadata"
                ),
                "kinematic_consistency_threshold": options.kinematic_consistency_threshold,
                "kinematic_consistency_policy": (
                    "drop local frame window around inconsistent full-coordinate q/dq steps, including base "
                    "position, base orientation, joint positions, and integrated velocity jumps"
                ),
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
                contact_detail_sensors=tuple(sensor for _name, sensor in self.contact_detail_sensors),
                contact_body_names=self.contact_body_names,
                tendon_manager=self.tendon_manager,
                joint_acc_for_inertia=joint_acc_recording,
                root_acc_for_inertia=root_acc_recording,
                ddq_source=self.recorder.cfg.ddq_source,
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
            ddq_override={"joint_acc": joint_acc_recording, "root_acc": root_acc_recording},
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

        tau = _joint_to_full_generalized(self.robot, motor_torque_tensor(self.robot))
        active_contact_sensors = tuple(sensor for _name, sensor in self.contact_detail_sensors) or (
            (self.contact_sensor,) if self.contact_sensor is not None else ()
        )
        if active_contact_sensors:
            contact_components = _projected_contact_sensors_components(
                self.robot, active_contact_sensors, self.contact_body_names
            )
            contact_force = contact_components["contact_force"]
            contact_moment = contact_components["contact_moment"]
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
        joint_position_error = current_pos - previous_pos - 0.5 * (previous_vel + current_vel) * dt
        joint_velocity_step_error = (current_vel - previous_vel) * dt

        current_root_pos = self.robot.data.root_pos_w
        current_root_quat = self.robot.data.root_quat_w
        current_root_vel = self.robot.data.root_com_vel_w
        previous_root_pos = self._previous_kinematic_root_pos
        previous_root_quat = self._previous_kinematic_root_quat
        previous_root_vel = self._previous_kinematic_root_vel
        root_position_error = (
            current_root_pos - previous_root_pos - 0.5 * (previous_root_vel[:, :3] + current_root_vel[:, :3]) * dt
        )
        root_angular_delta = 0.5 * (previous_root_vel[:, 3:6] + current_root_vel[:, 3:6]) * dt
        predicted_root_quat = quat_box_plus(previous_root_quat, root_angular_delta)
        root_orientation_error = quat_error_magnitude(current_root_quat, predicted_root_quat)
        root_velocity_step_error = (current_root_vel - previous_root_vel) * dt

        max_error = torch.maximum(joint_position_error.abs().amax(dim=1), joint_velocity_step_error.abs().amax(dim=1))
        max_error = torch.maximum(max_error, root_position_error.abs().amax(dim=1))
        max_error = torch.maximum(max_error, root_orientation_error)
        max_error = torch.maximum(max_error, root_velocity_step_error.abs().amax(dim=1))
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
        self._previous_kinematic_root_pos = self.robot.data.root_pos_w.clone()
        self._previous_kinematic_root_quat = self.robot.data.root_quat_w.clone()
        self._previous_kinematic_root_vel = self.robot.data.root_com_vel_w.clone()
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


def _find_contact_detail_sensors(env, primary_sensor_name: str) -> tuple[tuple[str, ContactSensor], ...]:
    sensors = getattr(env.scene, "sensors", {})
    prefix = f"{primary_sensor_name}_detail_"
    return tuple(
        (name, sensor)
        for name, sensor in sorted(sensors.items())
        if name.startswith(prefix) and isinstance(sensor, ContactSensor)
    )


def _recording_contact_body_names(
    params: ForrestParameterConfig,
    *,
    primary_sensor: ContactSensor | None,
    detail_sensors: tuple[tuple[str, ContactSensor], ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for source_names in (
        params.training.contacts.contact_sensor_body_names,
        params.physics.articulation.selective_self_collision_body_names,
        tuple(primary_sensor.body_names) if primary_sensor is not None else (),
        tuple(name for _sensor_name, sensor in detail_sensors for name in sensor.body_names),
    ):
        for body_name in source_names:
            if body_name not in names:
                names.append(body_name)
    return tuple(names)


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
    if side == "full":
        return "full_robot"
    raise ValueError(f"Unsupported recording side: {side}")


def _actuated_joint_indices(robot) -> list[int]:
    nonzero = torch.any(robot.data.default_joint_stiffness != 0.0, dim=0)
    return [int(index) for index in torch.nonzero(nonzero, as_tuple=False).squeeze(-1).tolist()]


def _tau_source_tensor(robot, tau_source: str):
    if tau_source == "actuation_command":
        return actuation_command_tensor(robot)
    if tau_source == "motor_torque":
        return motor_torque_tensor(robot)
    if tau_source == "applied_torque":
        return robot.data.applied_torque
    if tau_source == "computed_torque":
        return robot.data.computed_torque
    if tau_source == "zero":
        return robot.data.joint_pos * 0.0
    raise ValueError(f"Unsupported RL recording tau source: {tau_source}")


def _full_generalized_width(robot) -> int:
    return BASE_COORDINATE_COUNT + int(robot.data.joint_pos.shape[1])


def _joint_to_full_generalized(robot, joint_values: torch.Tensor) -> torch.Tensor:
    base = joint_values.new_zeros((joint_values.shape[0], BASE_COORDINATE_COUNT))
    return torch.cat((base, joint_values), dim=-1)


def _base_only_generalized(values: torch.Tensor) -> torch.Tensor:
    masked = torch.zeros_like(values)
    masked[:, :BASE_COORDINATE_COUNT] = values[:, :BASE_COORDINATE_COUNT]
    return masked


def _full_generalized_acceleration(robot, *, joint_acc: torch.Tensor, root_acc: torch.Tensor | None) -> torch.Tensor:
    if root_acc is None:
        root_acc = torch.zeros(
            (joint_acc.shape[0], BASE_COORDINATE_COUNT), dtype=joint_acc.dtype, device=joint_acc.device
        )
    return torch.cat((root_acc, joint_acc), dim=-1)


def _full_generalized_mass_matrix(robot, mass_matrices_full: torch.Tensor) -> torch.Tensor:
    joint_count = int(robot.data.joint_pos.shape[1])
    full_count = BASE_COORDINATE_COUNT + joint_count
    if mass_matrices_full.shape[-1] == full_count:
        return mass_matrices_full
    if mass_matrices_full.shape[-1] != joint_count:
        raise RuntimeError(
            f"Unexpected generalized mass matrix shape {tuple(mass_matrices_full.shape)} for {joint_count} joints."
        )
    mass_matrix = mass_matrices_full.new_zeros((mass_matrices_full.shape[0], full_count, full_count))
    mass_matrix[:, BASE_COORDINATE_COUNT:, BASE_COORDINATE_COUNT:] = mass_matrices_full
    return mass_matrix


def _project_body_wrenches_to_generalized(
    robot,
    *,
    body_indices: list[int],
    forces_w: torch.Tensor,
    torques_w: torch.Tensor,
) -> torch.Tensor:
    num_joints = int(robot.data.joint_pos.shape[1])
    full_width = _full_generalized_width(robot)
    tau = torch.zeros((robot.num_instances, full_width), dtype=robot.data.joint_pos.dtype, device=robot.device)
    if not body_indices:
        return tau

    if robot.is_fixed_base:
        jacobian_column_ids = list(range(num_joints))
        target_offset = BASE_COORDINATE_COUNT
    else:
        jacobian_column_ids = list(range(full_width))
        target_offset = 0
    jacobians = robot.root_physx_view.get_jacobians()

    for local_body_index, body_index in enumerate(body_indices):
        jacobian_body_index = int(body_index) - 1 if robot.is_fixed_base else int(body_index)
        jacobian_linear = jacobians[:, jacobian_body_index, 0:3, :][:, :, jacobian_column_ids]
        jacobian_angular = jacobians[:, jacobian_body_index, 3:6, :][:, :, jacobian_column_ids]
        force = forces_w[:, local_body_index, :]
        torque = torques_w[:, local_body_index, :]
        projected = torch.bmm(jacobian_linear.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
        projected += torch.bmm(jacobian_angular.transpose(1, 2), torque.unsqueeze(-1)).squeeze(-1)
        tau[:, target_offset : target_offset + projected.shape[1]] += projected
    return tau


def _projected_contact_sensor_torque_parts(robot, contact_sensor: ContactSensor, contact_body_names: tuple[str, ...]):
    components = _projected_contact_sensor_components(robot, contact_sensor, contact_body_names)
    return components["contact_force"], components["contact_moment"]


def _zero_generalized(robot) -> torch.Tensor:
    return torch.zeros((robot.num_instances, _full_generalized_width(robot)), device=robot.device)


def _zero_contact_components(robot) -> dict[str, torch.Tensor]:
    zero = _zero_generalized(robot)
    return {
        "contact_force": zero,
        "contact_moment": zero.clone(),
        "contact_normal": zero.clone(),
        "contact_friction": zero.clone(),
        "contact": zero.clone(),
    }


def _projected_contact_sensor_components(
    robot,
    contact_sensor: ContactSensor,
    contact_body_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    zero_components = _zero_contact_components(robot)
    available_names = set(contact_sensor.body_names)
    selected_names = tuple(name for name in contact_body_names if name in available_names)
    if not selected_names:
        return zero_components

    sensor_body_indices = [contact_sensor.body_names.index(name) for name in selected_names]
    robot_body_indices, _ = robot.find_bodies(list(selected_names), preserve_order=True)

    if contact_sensor.data.force_matrix_w is not None:
        normal_forces_by_filter = contact_sensor.data.force_matrix_w[:, sensor_body_indices, :, :]
    else:
        normal_forces_by_filter = contact_sensor.data.net_forces_w[:, sensor_body_indices, :].unsqueeze(2)
    normal_forces_by_filter = torch.nan_to_num(normal_forces_by_filter, nan=0.0)
    friction_forces = getattr(contact_sensor.data, "friction_forces_w", None)
    if friction_forces is not None:
        friction_forces_by_filter = torch.nan_to_num(friction_forces[:, sensor_body_indices, :, :], nan=0.0)
    else:
        friction_forces_by_filter = torch.zeros_like(normal_forces_by_filter)
    normal_forces_world = normal_forces_by_filter.sum(dim=2)
    friction_forces_world = friction_forces_by_filter.sum(dim=2)
    forces_world = normal_forces_world + friction_forces_world
    if not torch.any(forces_world).item():
        return zero_components

    contact_pos = getattr(contact_sensor.data, "contact_pos_w", None)
    if contact_pos is not None:
        selected_contact_pos = contact_pos[:, sensor_body_indices, :, :]
        valid_contact = torch.isfinite(selected_contact_pos).all(dim=-1, keepdim=True)
        contact_pos_world = torch.nan_to_num(selected_contact_pos, nan=0.0)
    else:
        contact_pos_world = None

    normal_torques = torch.zeros(
        (robot.num_instances, len(robot_body_indices), 3),
        dtype=forces_world.dtype,
        device=forces_world.device,
    )
    friction_torques = torch.zeros(
        (robot.num_instances, len(robot_body_indices), 3),
        dtype=forces_world.dtype,
        device=forces_world.device,
    )
    for local_foot_index, body_index in enumerate(robot_body_indices):
        if contact_pos_world is not None:
            body_pos = robot.data.body_pos_w[:, int(body_index), :]
            lever = contact_pos_world[:, local_foot_index, :, :] - body_pos.unsqueeze(1)
            # Match the contact sensor force sign to the angular Jacobian wrench convention.
            normal_moment_by_filter = torch.cross(normal_forces_by_filter[:, local_foot_index, :, :], lever, dim=-1)
            friction_moment_by_filter = torch.cross(friction_forces_by_filter[:, local_foot_index, :, :], lever, dim=-1)
            normal_torques[:, local_foot_index, :] = torch.where(
                valid_contact[:, local_foot_index, :, :],
                normal_moment_by_filter,
                torch.zeros_like(normal_moment_by_filter),
            ).sum(dim=1)
            friction_torques[:, local_foot_index, :] = torch.where(
                valid_contact[:, local_foot_index, :, :],
                friction_moment_by_filter,
                torch.zeros_like(friction_moment_by_filter),
            ).sum(dim=1)
    normal_force_tau = _project_body_wrenches_to_generalized(
        robot,
        body_indices=[int(index) for index in robot_body_indices],
        forces_w=normal_forces_world,
        torques_w=torch.zeros_like(normal_torques),
    )
    friction_force_tau = _project_body_wrenches_to_generalized(
        robot,
        body_indices=[int(index) for index in robot_body_indices],
        forces_w=friction_forces_world,
        torques_w=torch.zeros_like(friction_torques),
    )
    normal_moment_tau = _project_body_wrenches_to_generalized(
        robot,
        body_indices=[int(index) for index in robot_body_indices],
        forces_w=torch.zeros_like(normal_forces_world),
        torques_w=normal_torques,
    )
    friction_moment_tau = _project_body_wrenches_to_generalized(
        robot,
        body_indices=[int(index) for index in robot_body_indices],
        forces_w=torch.zeros_like(friction_forces_world),
        torques_w=friction_torques,
    )
    contact_normal = normal_force_tau + normal_moment_tau
    contact_friction = friction_force_tau + friction_moment_tau
    contact_force = normal_force_tau + friction_force_tau
    contact_moment = normal_moment_tau + friction_moment_tau
    return {
        "contact_force": contact_force,
        "contact_moment": contact_moment,
        "contact_normal": contact_normal,
        "contact_friction": contact_friction,
        "contact": contact_normal + contact_friction,
    }


def _projected_contact_sensors_components(
    robot,
    contact_sensors: tuple[ContactSensor, ...],
    contact_body_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    summed = _zero_contact_components(robot)
    if not contact_sensors:
        return summed
    for contact_sensor in contact_sensors:
        components = _projected_contact_sensor_components(robot, contact_sensor, contact_body_names)
        for name, values in components.items():
            summed[name] = summed[name] + values
    return summed


def _projected_contact_sensor_torque_groups(
    robot,
    contact_sensors: tuple[ContactSensor, ...],
    contact_body_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    grouped: dict[str, torch.Tensor] = {}
    for group_name, patterns in CONTACT_GROUP_PATTERNS.items():
        names = tuple(name for name in contact_body_names if any(pattern in name for pattern in patterns))
        components = _projected_contact_sensors_components(robot, contact_sensors, names)
        grouped[f"contact_{group_name}_force"] = components["contact_force"]
        grouped[f"contact_{group_name}_moment"] = components["contact_moment"]
        grouped[f"contact_{group_name}"] = components["contact"]
    return grouped


def _zero_contact_sensor_torque_groups(robot) -> dict[str, torch.Tensor]:
    grouped: dict[str, torch.Tensor] = {}
    for group_name in CONTACT_GROUP_PATTERNS:
        zero = torch.zeros((robot.num_instances, _full_generalized_width(robot)), device=robot.device)
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
    contact_detail_sensors: tuple[ContactSensor, ...] = (),
    contact_body_names: tuple[str, ...] = (),
    tendon_manager=None,
    joint_acc_for_inertia=None,
    root_acc_for_inertia=None,
    ddq_source: str = "physx_raw",
    include_debug: bool = False,
) -> dict[str, Any]:
    mass_matrices_full = robot.root_physx_view.get_generalized_mass_matrices()
    mass_matrices = _full_generalized_mass_matrix(robot, mass_matrices_full)
    raw_joint_acc = robot.data.joint_acc
    recording_interval_joint_acc = raw_joint_acc if joint_acc_for_inertia is None else joint_acc_for_inertia
    if robot.is_fixed_base:
        root_acc_raw = torch.zeros(
            (robot.num_instances, BASE_COORDINATE_COUNT), dtype=raw_joint_acc.dtype, device=raw_joint_acc.device
        )
    else:
        link_accelerations = robot.root_physx_view.get_link_accelerations()
        root_acc_raw = link_accelerations[:, 0, :]
    recording_interval_root_acc = root_acc_raw if root_acc_for_inertia is None else root_acc_for_inertia
    generalized_acc = _full_generalized_acceleration(
        robot,
        joint_acc=recording_interval_joint_acc,
        root_acc=recording_interval_root_acc,
    )
    generalized_acc_raw = _full_generalized_acceleration(robot, joint_acc=raw_joint_acc, root_acc=root_acc_raw)
    inertia_recording_interval = torch.bmm(mass_matrices, generalized_acc.unsqueeze(-1)).squeeze(-1)
    inertia_full_raw = torch.bmm(mass_matrices, generalized_acc_raw.unsqueeze(-1)).squeeze(-1)
    inertia_joint_all = torch.bmm(
        mass_matrices[:, :, BASE_COORDINATE_COUNT:],
        raw_joint_acc.unsqueeze(-1),
    ).squeeze(-1)
    inertia_joint_only = _joint_to_full_generalized(
        robot,
        torch.bmm(
            mass_matrices[:, BASE_COORDINATE_COUNT:, BASE_COORDINATE_COUNT:],
            raw_joint_acc.unsqueeze(-1),
        ).squeeze(-1),
    )
    inertia_leg_self = torch.zeros_like(inertia_full_raw)
    inertia_other_joints = torch.zeros_like(inertia_full_raw)
    inertia_raw = inertia_full_raw
    inertia_root_coupling_raw = inertia_full_raw - inertia_joint_all
    inertia_root_coupling = inertia_recording_interval - torch.bmm(
        mass_matrices[:, :, BASE_COORDINATE_COUNT:],
        recording_interval_joint_acc.unsqueeze(-1),
    ).squeeze(-1)
    inertia_root_coupling_alt = inertia_root_coupling_raw
    inertia_root_coupled_alt = inertia_full_raw
    if ddq_source == "physx_raw":
        inertia = inertia_raw
    elif ddq_source == "recording_interval":
        inertia = inertia_recording_interval
    else:
        raise ValueError(f"Unsupported ddq_source: {ddq_source!r}")
    coriolis, coriolis_force_api, coriolis_compensation_actual = _full_generalized_force_variants(
        robot,
        force_api_name="get_coriolis_and_centrifugal_forces",
        compensation_api_name="get_coriolis_and_centrifugal_compensation_forces",
        compensation_fallback_sign=1.0,
    )
    gravity, gravity_force_api, gravity_compensation_actual = _full_generalized_force_variants(
        robot,
        force_api_name="get_generalized_gravity_forces",
        compensation_api_name="get_gravity_compensation_forces",
        compensation_fallback_sign=-1.0,
    )
    coriolis_api_delta = coriolis_force_api - coriolis_compensation_actual
    gravity_api_delta = gravity_force_api - gravity_compensation_actual
    external_base_gravity = _base_only_generalized(gravity_compensation_actual)
    gravity_identification = gravity - external_base_gravity

    dynamic = robot.data.joint_dynamic_friction_coeff
    viscous = robot.data.joint_viscous_friction_coeff
    friction_dynamic = -dynamic * torch.sign(robot.data.joint_vel)
    friction_viscous = -viscous * robot.data.joint_vel
    friction = _joint_to_full_generalized(robot, friction_dynamic + friction_viscous)
    friction_dynamic = _joint_to_full_generalized(robot, friction_dynamic)
    friction_viscous = _joint_to_full_generalized(robot, friction_viscous)
    pantograph_spring = _joint_to_full_generalized(robot, _pantograph_spring_torque(robot))
    pantograph_damping = _joint_to_full_generalized(robot, _pantograph_damping_torque(robot))
    pantograph_applied_actuation = _joint_to_full_generalized(robot, _pantograph_actuation_torque(robot))
    pantograph_actuation = pantograph_applied_actuation
    motor_torque = motor_torque_tensor(robot)
    motor_actuation = _joint_to_full_generalized(robot, motor_torque)
    knee_flexor_actuation = _joint_to_full_generalized(
        robot,
        _knee_flexor_actuation_torque(robot, motor_torque),
    )
    actuation_command = _joint_to_full_generalized(robot, actuation_command_tensor(robot))
    implicit_drive_joint, implicit_drive_saturation_joint = _implicit_drive_estimate_torque(robot)
    implicit_drive_estimate = _joint_to_full_generalized(robot, implicit_drive_joint)
    implicit_drive_saturation = _joint_to_full_generalized(robot, implicit_drive_saturation_joint)
    active_contact_sensors = contact_detail_sensors or ((contact_sensor,) if contact_sensor is not None else ())
    contact_components = _projected_contact_sensors_components(robot, active_contact_sensors, contact_body_names)
    contact_force = contact_components["contact_force"]
    contact_moment = contact_components["contact_moment"]
    contact_normal = contact_components["contact_normal"]
    contact_friction = contact_components["contact_friction"]
    contact = contact_components["contact"]
    contact_force_norm = torch.linalg.norm(contact_force, dim=-1, keepdim=True)
    contact_moment_norm = torch.linalg.norm(contact_moment, dim=-1, keepdim=True)
    contact_moment_valid = contact_moment_norm <= 2.0 * torch.clamp(contact_force_norm, min=1.0e-6)
    contact_validated = contact_force + torch.where(
        contact_moment_valid, contact_moment, torch.zeros_like(contact_moment)
    )
    contact_identification = contact_validated
    tendon = _projected_tendon_wrench_torque(robot, tendon_manager)
    tendon_model = _joint_to_full_generalized(robot, _tendon_joint_torque_tensor(robot, tendon_manager))
    tendon_projection_delta = tendon - tendon_model
    permanent_wrench_total = _projected_permanent_wrench_torque(robot)
    solver_joint = _joint_to_full_generalized(robot, robot.root_physx_view.get_dof_projected_joint_forces())
    joint_limit_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    joint_limit_upper = robot.data.soft_joint_pos_limits[:, :, 1]
    joint_limit_distance_lower = robot.data.joint_pos - joint_limit_lower
    joint_limit_distance_upper = joint_limit_upper - robot.data.joint_pos
    joint_limit_distance_min = torch.minimum(joint_limit_distance_lower, joint_limit_distance_upper)
    solver_constraint_passive = _joint_to_full_generalized(
        robot,
        _passive_solver_constraint(robot, solver_joint[:, BASE_COORDINATE_COUNT:]),
    )
    solver_constraint_limit = torch.where(
        _joint_to_full_generalized(robot, joint_limit_distance_min <= 0.05),
        solver_joint,
        torch.zeros_like(solver_joint),
    )
    solver_constraint_internal = torch.where(
        (solver_constraint_passive != 0.0) | (solver_constraint_limit != 0.0),
        solver_joint,
        torch.zeros_like(solver_joint),
    )
    conservative = inertia + gravity_identification + coriolis + tendon
    non_conservative = actuation_command + contact_identification + friction
    residual = conservative - non_conservative
    if not include_debug:
        return {
            "inertia": inertia,
            "inertia_recording_interval": inertia_recording_interval,
            "inertia_raw": inertia_raw,
            "inertia_joint_only": inertia_joint_only,
            "inertia_joint_all": inertia_joint_all,
            "inertia_root_coupling": inertia_root_coupling,
            "inertia_root_coupling_raw": inertia_root_coupling_raw,
            "coriolis": coriolis,
            "coriolis_force_api": coriolis_force_api,
            "coriolis_compensation_actual": coriolis_compensation_actual,
            "coriolis_api_delta": coriolis_api_delta,
            "gravity": gravity,
            "gravity_identification": gravity_identification,
            "gravity_force_api": gravity_force_api,
            "gravity_compensation_actual": gravity_compensation_actual,
            "gravity_api_delta": gravity_api_delta,
            "external_base_gravity": external_base_gravity,
            "friction": friction,
            "motor_actuation": motor_actuation,
            "knee_flexor_actuation": knee_flexor_actuation,
            "implicit_drive_estimate": implicit_drive_estimate,
            "implicit_drive_saturation": implicit_drive_saturation,
            "actuation_command": actuation_command,
            "contact": contact,
            "contact_force": contact_force,
            "contact_moment": contact_moment,
            "contact_normal": contact_normal,
            "contact_friction": contact_friction,
            "contact_validated": contact_validated,
            "contact_identification": contact_identification,
            "tendon": tendon,
            "permanent_wrench_total": permanent_wrench_total,
            "solver_constraint_internal": solver_constraint_internal,
            "residual": residual,
            "pantograph_spring": pantograph_spring,
            "pantograph_damping": pantograph_damping,
            "pantograph_actuation": pantograph_actuation,
            "pantograph_applied_actuation": pantograph_applied_actuation,
            "joint_acc_for_inertia": raw_joint_acc,
            "root_acc_for_inertia": root_acc_raw,
            "joint_acc_recording_interval": recording_interval_joint_acc,
            "root_acc_recording_interval": recording_interval_root_acc,
        }

    generalized_acc_physx_base_recording_joints = _full_generalized_acceleration(
        robot,
        joint_acc=recording_interval_joint_acc,
        root_acc=root_acc_raw,
    )
    generalized_acc_recording_base_physx_joints = _full_generalized_acceleration(
        robot,
        joint_acc=raw_joint_acc,
        root_acc=recording_interval_root_acc,
    )
    root_acc_raw_body = torch.cat(
        (
            quat_apply_inverse(robot.data.root_quat_w, root_acc_raw[:, :3]),
            quat_apply_inverse(robot.data.root_quat_w, root_acc_raw[:, 3:6]),
        ),
        dim=-1,
    )
    root_acc_raw_swapped = torch.cat((root_acc_raw[:, 3:6], root_acc_raw[:, :3]), dim=-1)
    generalized_acc_physx_base_body_frame = _full_generalized_acceleration(
        robot,
        joint_acc=raw_joint_acc,
        root_acc=root_acc_raw_body,
    )
    generalized_acc_physx_base_swapped = _full_generalized_acceleration(
        robot,
        joint_acc=raw_joint_acc,
        root_acc=root_acc_raw_swapped,
    )
    inertia_physx_base_recording_joints = torch.bmm(
        mass_matrices,
        generalized_acc_physx_base_recording_joints.unsqueeze(-1),
    ).squeeze(-1)
    inertia_recording_base_physx_joints = torch.bmm(
        mass_matrices,
        generalized_acc_recording_base_physx_joints.unsqueeze(-1),
    ).squeeze(-1)
    inertia_physx_base_body_frame = torch.bmm(
        mass_matrices,
        generalized_acc_physx_base_body_frame.unsqueeze(-1),
    ).squeeze(-1)
    inertia_physx_base_swapped = torch.bmm(
        mass_matrices,
        generalized_acc_physx_base_swapped.unsqueeze(-1),
    ).squeeze(-1)

    physx_actuation = _joint_to_full_generalized(robot, robot.root_physx_view.get_dof_actuation_forces())
    pantograph_computed_actuation = _joint_to_full_generalized(robot, _pantograph_computed_actuation_torque(robot))
    pantograph_reconstructed_actuation = _joint_to_full_generalized(
        robot,
        _pantograph_reconstructed_actuation_torque(
            robot,
            pantograph_spring=pantograph_spring[:, BASE_COORDINATE_COUNT:],
            pantograph_damping=pantograph_damping[:, BASE_COORDINATE_COUNT:],
        ),
    )
    pantograph_actuation_error = pantograph_applied_actuation - pantograph_reconstructed_actuation
    actuation = actuation_command
    if active_contact_sensors:
        contact_groups = _projected_contact_sensor_torque_groups(robot, active_contact_sensors, contact_body_names)
    else:
        contact_groups = _zero_contact_sensor_torque_groups(robot)
    with_pantograph_actuation = actuation_command + pantograph_actuation
    with_knee_flexor_actuation = actuation_command + knee_flexor_actuation
    with_pantograph_and_knee_flexor_actuation = actuation_command + pantograph_actuation + knee_flexor_actuation
    residual_with_pantograph_actuation = conservative - (with_pantograph_actuation + contact_identification + friction)
    residual_with_knee_flexor_actuation = conservative - (
        with_knee_flexor_actuation + contact_identification + friction
    )
    residual_with_pantograph_and_knee_flexor_actuation = conservative - (
        with_pantograph_and_knee_flexor_actuation + contact_identification + friction
    )
    residual_no_pantograph_actuation = residual
    residual_no_knee_flexor_actuation = residual
    residual_no_pantograph_no_knee_flexor_actuation = residual
    residual_no_pantograph_no_knee_flexor_plus_solver = conservative - (
        actuation_command + contact_identification + friction + solver_constraint_internal
    )

    return {
        "inertia": inertia,
        "inertia_recording_interval": inertia_recording_interval,
        "inertia_raw": inertia_raw,
        "inertia_physx_base_recording_joints": inertia_physx_base_recording_joints,
        "inertia_recording_base_physx_joints": inertia_recording_base_physx_joints,
        "inertia_physx_base_body_frame": inertia_physx_base_body_frame,
        "inertia_physx_base_swapped": inertia_physx_base_swapped,
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
        "coriolis_force_api": coriolis_force_api,
        "coriolis_compensation_actual": coriolis_compensation_actual,
        "coriolis_api_delta": coriolis_api_delta,
        "gravity": gravity,
        "gravity_identification": gravity_identification,
        "gravity_force_api": gravity_force_api,
        "gravity_compensation_actual": gravity_compensation_actual,
        "gravity_api_delta": gravity_api_delta,
        "external_base_gravity": external_base_gravity,
        "friction_dynamic": friction_dynamic,
        "friction_viscous": friction_viscous,
        "friction": friction,
        "motor_actuation": motor_actuation,
        "knee_flexor_actuation": knee_flexor_actuation,
        "implicit_drive_estimate": implicit_drive_estimate,
        "implicit_drive_saturation": implicit_drive_saturation,
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
        "contact_normal": contact_normal,
        "contact_friction": contact_friction,
        "contact_validated": contact_validated,
        "contact_identification": contact_identification,
        **contact_groups,
        "tendon": tendon,
        "permanent_wrench_total": permanent_wrench_total,
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
        "joint_acc_for_inertia": raw_joint_acc,
        "root_acc_for_inertia": root_acc_raw,
        "joint_acc_recording_interval": recording_interval_joint_acc,
        "root_acc_recording_interval": recording_interval_root_acc,
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


def _implicit_drive_estimate_torque(robot) -> tuple[torch.Tensor, torch.Tensor]:
    computed = (
        robot.data.joint_stiffness * (robot.data.joint_pos_target - robot.data.joint_pos)
        + robot.data.joint_damping * (robot.data.joint_vel_target - robot.data.joint_vel)
        + robot.data.joint_effort_target
    )
    effort_limits = torch.nan_to_num(robot.data.joint_effort_limits, nan=float("inf"), posinf=float("inf"))
    clipped = torch.clamp(computed, min=-effort_limits, max=effort_limits)
    return clipped, computed - clipped


def _projected_permanent_wrench_torque(robot) -> torch.Tensor:
    composer = getattr(robot, "permanent_wrench_composer", None)
    if composer is None or not getattr(composer, "active", False):
        return torch.zeros((robot.num_instances, _full_generalized_width(robot)), device=robot.device)

    forces_body = composer.composed_force_as_torch
    torques_body = composer.composed_torque_as_torch
    if forces_body.numel() == 0 and torques_body.numel() == 0:
        return torch.zeros((robot.num_instances, _full_generalized_width(robot)), device=robot.device)

    body_indices = list(range(int(robot.num_bodies)))
    forces_world = torch.stack(
        [
            quat_apply(robot.data.body_quat_w[:, body_index, :], forces_body[:, body_index, :])
            for body_index in body_indices
        ],
        dim=1,
    )
    torques_world = torch.stack(
        [
            quat_apply(robot.data.body_quat_w[:, body_index, :], torques_body[:, body_index, :])
            for body_index in body_indices
        ],
        dim=1,
    )
    projected = _project_body_wrenches_to_generalized(
        robot,
        body_indices=body_indices,
        forces_w=forces_world,
        torques_w=torques_world,
    )
    return -projected


def _projected_tendon_wrench_torque(robot, tendon_manager):
    if tendon_manager is None:
        return torch.zeros((robot.num_instances, _full_generalized_width(robot)), device=robot.device)

    link_torques = getattr(tendon_manager, "cached_tendon_link_torques", None)
    link_forces = getattr(tendon_manager, "cached_tendon_forces", None)
    body_ids = getattr(tendon_manager, "cached_tendon_body_ids", None)
    if link_torques is None or body_ids is None:
        return _joint_to_full_generalized(robot, _tendon_joint_torque_tensor(robot, tendon_manager))

    body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=robot.device).flatten()
    if body_ids.numel() == 0:
        return torch.zeros((robot.num_instances, _full_generalized_width(robot)), device=robot.device)

    if link_forces is None:
        link_forces = torch.zeros_like(link_torques)

    body_indices = [int(body_id) for body_id in body_ids.tolist()]
    forces_world = torch.stack(
        [
            quat_apply(robot.data.body_quat_w[:, body_index, :], link_forces[:, local_body_index, :])
            for local_body_index, body_index in enumerate(body_indices)
        ],
        dim=1,
    )
    torques_world = torch.stack(
        [
            quat_apply(robot.data.body_quat_w[:, body_index, :], link_torques[:, local_body_index, :])
            for local_body_index, body_index in enumerate(body_indices)
        ],
        dim=1,
    )
    tau_tendon = _project_body_wrenches_to_generalized(
        robot,
        body_indices=body_indices,
        forces_w=forces_world,
        torques_w=torques_world,
    )

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
    full_force = _actual_full_generalized_force(
        robot,
        force_api_name=force_api_name,
        compensation_api_name=compensation_api_name,
    )
    full_indices = [BASE_COORDINATE_COUNT + int(joint_id) for joint_id in joint_ids]
    return full_force[:, full_indices]


def _actual_full_generalized_force(
    robot,
    *,
    force_api_name: str,
    compensation_api_name: str,
):
    selected, _, _ = _full_generalized_force_variants(
        robot,
        force_api_name=force_api_name,
        compensation_api_name=compensation_api_name,
    )
    return selected


def _full_generalized_force_variants(
    robot,
    *,
    force_api_name: str,
    compensation_api_name: str,
    compensation_fallback_sign: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.zeros(
        (robot.num_instances, _full_generalized_width(robot)),
        dtype=robot.data.joint_pos.dtype,
        device=robot.device,
    )
    force_api_alias = zero
    compensation_actual = zero
    force_api_available = False
    compensation_api_available = False
    try:
        force_api = getattr(robot.root_physx_view, force_api_name)()
        force_api_alias = _joint_only_generalized(robot, _normalize_full_generalized_force(robot, force_api))
        force_api_available = True
    except Exception:
        pass

    try:
        compensation = getattr(robot.root_physx_view, compensation_api_name)()
        compensation_generalized = _normalize_full_generalized_force(robot, compensation)
        compensation_actual = -compensation_generalized
        compensation_api_available = True
    except Exception:
        pass

    if force_api_available:
        return force_api_alias, force_api_alias, compensation_actual
    if compensation_api_available:
        compensation_alias = compensation_fallback_sign * _joint_only_generalized(robot, compensation_generalized)
        return compensation_alias, compensation_alias, compensation_actual
    return force_api_alias, force_api_alias, compensation_actual


def _joint_only_generalized(robot, values: torch.Tensor) -> torch.Tensor:
    joint_only = torch.zeros_like(values)
    joint_count = int(robot.data.joint_pos.shape[1])
    joint_only[:, BASE_COORDINATE_COUNT : BASE_COORDINATE_COUNT + joint_count] = values[
        :, BASE_COORDINATE_COUNT : BASE_COORDINATE_COUNT + joint_count
    ]
    return joint_only


def _normalize_full_generalized_force(robot, force: torch.Tensor) -> torch.Tensor:
    joint_count = int(robot.data.joint_pos.shape[1])
    full_count = _full_generalized_width(robot)
    if force.shape[-1] == full_count:
        return force
    if force.shape[-1] == joint_count:
        return _joint_to_full_generalized(robot, force)
    raise RuntimeError(
        f"Unexpected generalized force shape {tuple(force.shape)}; expected width {joint_count} or {full_count}."
    )
