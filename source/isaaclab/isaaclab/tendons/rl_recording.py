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
        self.contact_body_names = tuple(params.training.contacts.foot_body_names)
        self.actuated_joint_indices = _actuated_joint_indices(self.robot)

        output_dir = options.output_dir or _timestamped_output_dir(params.run.output_dir)
        self.recorder = DataRecording(
            DataRecordingConfig(
                output_dir=output_dir,
                sqlite_filename=params.recording.kinematics_db_filename,
                tendon_sqlite_filename=params.recording.tendons_db_filename,
                dynamics_sqlite_filename=params.recording.dynamics_db_filename,
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
        self.recorder.record_step(
            step_index=self.step_index,
            sim_time=sim_time,
            robot=self.robot,
            extra_context={"source": "rsl_rl_play"},
            tau_override=tau_input,
        )
        if self.recorder.cfg.record_dynamics:
            self.recorder.record_dynamics_step(
                step_index=self.step_index,
                sim_time=sim_time,
                robot=self.robot,
                dynamics_terms=_dynamics_terms(self.robot),
                tau_input=tau_input,
            )
        return self.options.max_steps is None or self.step_index < self.options.max_steps

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
            tau += _projected_contact_sensor_torque(self.robot, self.contact_sensor, self.contact_body_names)
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


def _projected_contact_sensor_torque(robot, contact_sensor: ContactSensor, contact_body_names: tuple[str, ...]):
    tau_ground = torch.zeros_like(robot.data.joint_pos)
    available_names = set(contact_sensor.body_names)
    selected_names = tuple(name for name in contact_body_names if name in available_names)
    if not selected_names:
        return tau_ground

    sensor_body_indices = [contact_sensor.body_names.index(name) for name in selected_names]
    robot_body_indices, _ = robot.find_bodies(list(selected_names), preserve_order=True)

    forces_world = contact_sensor.data.net_forces_w[:, sensor_body_indices, :]
    if not torch.any(forces_world).item():
        return tau_ground

    num_joints = robot.data.joint_pos.shape[1]
    joint_ids = list(range(num_joints))
    jacobian_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    jacobians = robot.root_physx_view.get_jacobians()

    for local_foot_index, body_index in enumerate(robot_body_indices):
        jacobian_body_index = int(body_index) - 1 if robot.is_fixed_base else int(body_index)
        jacobian_pos = jacobians[:, jacobian_body_index, 0:3, :][:, :, jacobian_joint_ids]
        force = forces_world[:, local_foot_index, :].unsqueeze(-1)
        tau_ground += torch.bmm(jacobian_pos.transpose(1, 2), force).squeeze(-1)
    return tau_ground


def _dynamics_terms(robot) -> dict[str, Any]:
    num_joints = robot.data.joint_pos.shape[1]
    joint_ids = list(range(num_joints))
    generalized_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]

    mass_matrices = robot.root_physx_view.get_generalized_mass_matrices()
    if not robot.is_fixed_base:
        mass_matrices = mass_matrices[:, 6:, 6:]
    inertia = torch.bmm(mass_matrices, robot.data.joint_acc.unsqueeze(-1)).squeeze(-1)

    coriolis_all = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
    gravity_all = robot.root_physx_view.get_gravity_compensation_forces()
    dynamic = robot.data.joint_dynamic_friction_coeff
    viscous = robot.data.joint_viscous_friction_coeff

    return {
        "inertia": inertia,
        "coriolis": coriolis_all[:, generalized_joint_ids],
        "gravity": gravity_all[:, generalized_joint_ids],
        "friction": -dynamic * torch.sign(robot.data.joint_vel) - viscous * robot.data.joint_vel,
    }
