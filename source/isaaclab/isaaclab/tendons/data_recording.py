# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identix-compatible data recording helpers for Forrest tendon simulations."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from isaaclab.tendons.models.analytic.constants import (
    actuated_joint_names,
    joint_names_left,
    joint_names_right,
    link_names_left,
    link_names_right,
)
from isaaclab.utils.math import euler_xyz_from_quat

TENDON_CHAIN_5_JOINTS: dict[str, tuple[str, ...]] = {
    "left": tuple(joint_names_left),
    "right": tuple(joint_names_right),
}
REAL_LEG_JOINTS: dict[str, tuple[str, ...]] = {
    "left": (
        "l0_acetabulofemoral_roll",
        "l1_acetabulofemoral_lateral",
        "lp1_pantograph",
        "l2_pseudo_acetabulofemoral_flexion",
        "l3b_femorotibial_back",
        "l3f_femorotibial_front",
        "l4f_intertarsal_front",
        "l4b_intertarsal_back",
        "l4p_intertarsal_pulley",
        "l5_metatarsophalangeal",
        "l6_interphalangeal",
        "l8_knee_flexor",
    ),
    "right": (
        "r0_acetabulofemoral_roll",
        "r1_acetabulofemoral_lateral",
        "rp1_pantograph",
        "r2_pseudo_acetabulofemoral_flexion",
        "r3b_femorotibial_back",
        "r3f_femorotibial_front",
        "r4f_intertarsal_front",
        "r4b_intertarsal_back",
        "r4p_intertarsal_pulley",
        "r5_metatarsophalangeal",
        "r6_interphalangeal",
        "r8_knee_flexor",
    ),
}
TENDON_CHAIN_LINKS: dict[str, tuple[str, ...]] = {
    "left": tuple(link_names_left),
    "right": tuple(link_names_right),
}

BASE_COORDINATE_NAMES = (
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_roll",
    "base_pitch",
    "base_yaw",
)
BASE_COORDINATE_UNITS = ("m", "m", "m", "rad", "rad", "rad")
BASE_VELOCITY_UNITS = ("m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s")
BASE_ACCELERATION_UNITS = ("m/s^2", "m/s^2", "m/s^2", "rad/s^2", "rad/s^2", "rad/s^2")
BASE_FORCE_UNITS = ("N", "N", "N", "N*m", "N*m", "N*m")
BASE_COORDINATE_COUNT = len(BASE_COORDINATE_NAMES)
FULL_ROBOT_STREAM = "full"

CONTACT_GROUP_NAMES = ("digit", "connector", "base", "self_collision")
KNEE_FLEXOR_JOINT_NAMES = ("l8_knee_flexor", "r8_knee_flexor")

TRAINING_DYNAMICS_TERM_NAMES = (
    "inertia",
    "coriolis",
    "gravity",
    "tendon",
    "actuation",
    "contact",
    "friction",
    "solver_constraint_internal",
    "residual",
    "external",
)

DEBUG_DYNAMICS_TERM_NAMES = (
    "inertia",
    "inertia_recording_interval",
    "inertia_raw",
    "inertia_physx_base_recording_joints",
    "inertia_recording_base_physx_joints",
    "inertia_physx_base_body_frame",
    "inertia_physx_base_swapped",
    "inertia_joint_only",
    "inertia_joint_all",
    "inertia_root_coupling",
    "inertia_root_coupling_raw",
    "coriolis",
    "coriolis_force_api",
    "coriolis_compensation_actual",
    "coriolis_api_delta",
    "gravity",
    "gravity_identification",
    "gravity_force_api",
    "gravity_compensation_actual",
    "gravity_api_delta",
    "external_base_gravity",
    "tendon",
    "pantograph_spring",
    "tendon_model",
    "tendon_projection_delta",
    "motor_actuation",
    "knee_flexor_actuation",
    "actuation",
    "actuation_command",
    "pantograph_actuation",
    "pantograph_applied_actuation",
    "pantograph_computed_actuation",
    "pantograph_reconstructed_actuation",
    "pantograph_actuation_error",
    "physx_actuation",
    "contact",
    "contact_force",
    "contact_moment",
    "contact_normal",
    "contact_friction",
    "contact_validated",
    "contact_identification",
    "contact_digit",
    "contact_digit_force",
    "contact_digit_moment",
    "contact_connector",
    "contact_connector_force",
    "contact_connector_moment",
    "contact_base",
    "contact_base_force",
    "contact_base_moment",
    "contact_self_collision",
    "friction",
    "implicit_drive_estimate",
    "permanent_wrench_total",
    "pantograph_damping",
    "solver_joint",
    "solver_constraint_internal",
    "residual",
    "residual_with_pantograph_actuation",
    "residual_with_knee_flexor_actuation",
    "residual_with_pantograph_and_knee_flexor_actuation",
    "residual_no_pantograph_actuation",
    "residual_no_knee_flexor_actuation",
    "residual_no_pantograph_no_knee_flexor_actuation",
    "residual_no_pantograph_no_knee_flexor_plus_solver",
)
DEBUG_DYNAMICS_MATRIX_TERM_NAMES = ("mass_matrix",)
DEBUG_DYNAMICS_SCALAR_NAMES = (
    "quality_residual_norm",
    "quality_contact_norm",
    "quality_inertia_norm",
    "quality_actuation_norm",
    "quality_contact_friction_norm",
    "quality_implicit_drive_norm",
    "quality_implicit_drive_saturation_norm",
    "quality_permanent_wrench_delta_norm",
)
SQLITE_DEBUG_COLUMN_LIMIT = 2000
OMITTED_JOINTS_BY_SET: dict[str, dict[str, tuple[str, ...]]] = {
    "real_leg_joints": {
        "left": (),
        "right": (),
    },
    "tendon_chain_5": {
        "left": (
            "lp1_pantograph",
            "l0_acetabulofemoral_roll",
            "l1_acetabulofemoral_lateral",
            "l2_pseudo_acetabulofemoral_flexion",
            "l3b_femorotibial_back",
            "l4b_intertarsal_back",
            "l4p_intertarsal_pulley",
        ),
        "right": (
            "rp1_pantograph",
            "r0_acetabulofemoral_roll",
            "r1_acetabulofemoral_lateral",
            "r2_pseudo_acetabulofemoral_flexion",
            "r3b_femorotibial_back",
            "r4b_intertarsal_back",
            "r4p_intertarsal_pulley",
        ),
    },
}


@dataclass
class DataRecordingConfig:
    """Configuration for one Identix-style recording run."""

    output_dir: str | Path
    sqlite_filename: str = "forrest_kinematics.db"
    tendon_sqlite_filename: str = "forrest_tendons.db"
    dynamics_sqlite_filename: str = "forrest_dynamics.db"
    debug_sqlite_filename: str = "debug.db"
    metadata_filename: str = "metadata.json"
    viz_vars_filename: str = "viz_vars.json"
    sim_table_name: str = "sim_data"
    joint_set: str = "real_leg_joints"
    side_policy: str = "left_only"
    selected_joint_names: tuple[str, ...] | None = None
    selected_env_ids: tuple[int, ...] | None = None
    body_set: str = "tendon_chain_links"
    selected_body_names: tuple[str, ...] | None = None
    record_base_state: bool = False
    record_spatial_state: bool = False
    sampling_stride: int = 1
    startup_skip_seconds: float = 0.0
    constraint_mode: str = "static"
    controller: str | None = "sin"
    tau_source: str = "actuation_command"
    ddq_source: str = "physx_raw"
    record_tendons: bool = True
    record_dynamics: bool = True
    record_debug_dynamics: bool = False
    residual_filter_threshold: float | None = None
    overwrite: bool = False
    batch_size: int = 512
    parameter_file: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class DataRecording:
    """Write Forrest simulation samples in Identix ``sim_data`` format.

    ``sim_data`` intentionally contains only positional Identix columns:
    ``q0..qN``, ``dq0..dqN``, ``ddq0..ddqN``, ``tau0..tauN``. Metadata such as
    joint names and simulation settings are stored in the sidecar JSON so the
    SQLite file stays compatible with Identix-style kinematics databases.
    """

    def __init__(self, cfg: DataRecordingConfig):
        if cfg.sampling_stride < 1:
            raise ValueError("sampling_stride must be >= 1.")
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        cfg.residual_filter_threshold = _normalize_optional_float(
            cfg.residual_filter_threshold,
            "residual_filter_threshold",
        )

        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.sqlite_path = self.output_dir / cfg.sqlite_filename
        self.tendon_sqlite_path = self.output_dir / cfg.tendon_sqlite_filename
        self.dynamics_sqlite_path = self.output_dir / cfg.dynamics_sqlite_filename
        self.debug_sqlite_path = self.output_dir / cfg.debug_sqlite_filename
        self.metadata_path = self.output_dir / cfg.metadata_filename
        self.viz_vars_path = self.output_dir / cfg.viz_vars_filename

        self._db: sqlite3.Connection | None = None
        self._tendon_db: sqlite3.Connection | None = None
        self._dynamics_db: sqlite3.Connection | None = None
        self._debug_db: sqlite3.Connection | None = None
        self._sim_rows_by_stream: dict[tuple[int, str], list[tuple[float, ...]]] = {}
        self._tendon_buffer: list[tuple[int, float, str, str]] = []
        self._dynamics_rows_by_stream: dict[tuple[int, str], list[tuple[Any, ...]]] = {}
        self._debug_rows_by_stream: dict[tuple[int, str], list[tuple[Any, ...]]] = {}
        self._joint_indices_by_side: dict[str, list[int]] = {}
        self._joint_names_by_side: dict[str, tuple[str, ...]] = {}
        self._base_coordinate_count_by_side: dict[str, int] = {}
        self._body_indices_by_side: dict[str, list[int]] = {}
        self._body_names_by_side: dict[str, tuple[str, ...]] = {}
        self._joint_dynamics_properties_rows: list[dict[str, Any]] = []
        self._body_dynamics_properties_rows: list[dict[str, Any]] = []
        self._sim_columns: list[str] = []
        self._dynamics_columns: list[str] = []
        self._debug_columns: list[str] = []
        self._debug_matrix_term_names: tuple[str, ...] = DEBUG_DYNAMICS_MATRIX_TERM_NAMES
        self._spatial_columns: list[str] = []
        self._row_count = 0
        self._tendon_row_count = 0
        self._dynamics_row_count = 0
        self._debug_row_count = 0
        self._next_dynamics_sample_id = 0
        self._next_debug_sample_id = 0
        self._initialized = False
        self._closed = False
        self._sim_dt: float | None = None
        self._context_metadata: dict[str, Any] = {}
        self._selected_env_ids: tuple[int, ...] = ()

    @property
    def num_dofs(self) -> int:
        return len(self._sim_columns) // 4

    def initialize(self, robot, *, sim_dt: float | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Resolve runtime indices and create output tables."""

        if self._initialized:
            raise RuntimeError("DataRecording is already initialized.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.cfg.overwrite:
            candidate_paths = [self.sqlite_path, self.metadata_path, self.viz_vars_path]
            if self.cfg.record_tendons:
                candidate_paths.append(self.tendon_sqlite_path)
            if self.cfg.record_dynamics:
                candidate_paths.append(self.dynamics_sqlite_path)
            if self.cfg.record_debug_dynamics:
                candidate_paths.append(self.debug_sqlite_path)
            existing = [path for path in candidate_paths if path.exists()]
            if existing:
                raise FileExistsError(f"Recording output already exists: {existing}")
        unlink_paths = [self.sqlite_path, self.metadata_path, self.viz_vars_path]
        if self.cfg.record_tendons:
            unlink_paths.append(self.tendon_sqlite_path)
        if self.cfg.record_dynamics:
            unlink_paths.append(self.dynamics_sqlite_path)
        if self.cfg.record_debug_dynamics:
            unlink_paths.append(self.debug_sqlite_path)
        for path in unlink_paths:
            if self.cfg.overwrite and path.exists():
                path.unlink()

        self._sim_dt = sim_dt
        self._context_metadata = dict(metadata or {})
        self._resolve_joint_indices(robot)
        self._resolve_env_ids(robot)
        self._resolve_body_indices(robot)
        self._resolve_joint_dynamics_properties(robot)
        self._resolve_body_dynamics_properties(robot)
        self._create_tables()
        self._maybe_print_base_constraint_warning()
        self._initialized = True

    def record_dynamics_step(
        self,
        *,
        step_index: int,
        sim_time: float,
        robot,
        dynamics_terms: dict[str, Any],
        tau_input,
        skip_env_ids: set[int] | None = None,
    ) -> None:
        """Record non-kinematic inverse-dynamics terms aligned to ``sim_data`` rows."""

        if not self.cfg.record_dynamics and not self.cfg.record_debug_dynamics:
            return
        if not self._initialized:
            raise RuntimeError("Call initialize(...) before record_dynamics_step(...).")
        if self._closed:
            raise RuntimeError("Cannot record after close().")
        if sim_time < self.cfg.startup_skip_seconds:
            return
        if step_index % self.cfg.sampling_stride != 0:
            return

        if self.cfg.record_debug_dynamics:
            debug_missing = [
                name
                for name in (*DEBUG_DYNAMICS_TERM_NAMES, *self._debug_matrix_term_names)
                if name not in dynamics_terms
            ]
            if debug_missing:
                raise ValueError(f"Missing debug dynamics term tensors: {debug_missing}")

        excluded_env_ids = set(skip_env_ids or ())
        for env_id in self._selected_env_ids:
            if int(env_id) in excluded_env_ids:
                continue
            for side in self._selected_sides():
                if not self._passes_residual_filter(
                    dynamics_terms=dynamics_terms,
                    env_id=int(env_id),
                    side=side,
                    robot=robot,
                ):
                    continue

                if self.cfg.record_dynamics:
                    row_values = [-1, int(step_index), float(sim_time), int(env_id), side]
                    for term_name in TRAINING_DYNAMICS_TERM_NAMES:
                        term_values = self._training_dynamics_term(
                            dynamics_terms=dynamics_terms,
                            term_name=term_name,
                            env_id=int(env_id),
                            side=side,
                            robot=robot,
                        )
                        row_values.extend(float(value) for value in term_values)
                    self._dynamics_rows_by_stream.setdefault((int(env_id), side), []).append(tuple(row_values))
                    self._dynamics_row_count += 1

                if self.cfg.record_debug_dynamics:
                    debug_values: list[Any] = [-1, int(step_index), float(sim_time), int(env_id), side]
                    for term_name in DEBUG_DYNAMICS_TERM_NAMES:
                        term_values = self._debug_dynamics_term(
                            dynamics_terms=dynamics_terms,
                            term_name=term_name,
                            env_id=int(env_id),
                            side=side,
                            robot=robot,
                        )
                        debug_values.extend(float(value) for value in term_values)
                    debug_values.extend(
                        self._dynamics_quality_scalars(
                            dynamics_terms=dynamics_terms,
                            env_id=int(env_id),
                            side=side,
                            robot=robot,
                        )[name]
                        for name in DEBUG_DYNAMICS_SCALAR_NAMES
                    )
                    for term_name in self._debug_matrix_term_names:
                        term_tensor = dynamics_terms[term_name][env_id]
                        selected = (
                            self._select_generalized_matrix(
                                term_tensor,
                                side=side,
                                robot=robot,
                            )
                            .detach()
                            .cpu()
                            .reshape(-1)
                            .tolist()
                        )
                        debug_values.extend(float(value) for value in selected)
                    self._debug_rows_by_stream.setdefault((int(env_id), side), []).append(tuple(debug_values))
                    self._debug_row_count += 1

    def _training_dynamics_term(
        self,
        *,
        dynamics_terms: dict[str, Any],
        term_name: str,
        env_id: int,
        side: str,
        robot,
    ) -> list[float]:
        if term_name == "inertia":
            values = self._select_generalized_vector(dynamics_terms["inertia"], env_id=env_id, side=side, robot=robot)
        elif term_name == "coriolis":
            values = self._select_generalized_vector(dynamics_terms["coriolis"], env_id=env_id, side=side, robot=robot)
        elif term_name == "gravity":
            values = self._select_generalized_vector(
                dynamics_terms.get("gravity_identification", dynamics_terms["gravity"]),
                env_id=env_id,
                side=side,
                robot=robot,
            )
        elif term_name == "tendon":
            values = self._select_generalized_vector(dynamics_terms["tendon"], env_id=env_id, side=side, robot=robot)
        elif term_name == "actuation":
            values = self._select_generalized_vector(
                dynamics_terms["actuation_command"], env_id=env_id, side=side, robot=robot
            )
        elif term_name == "contact":
            values = self._select_generalized_vector(
                dynamics_terms["contact_identification"], env_id=env_id, side=side, robot=robot
            )
        elif term_name == "friction":
            values = self._select_generalized_vector(dynamics_terms["friction"], env_id=env_id, side=side, robot=robot)
        elif term_name == "solver_constraint_internal":
            values = self._select_generalized_vector(
                dynamics_terms["solver_constraint_internal"], env_id=env_id, side=side, robot=robot
            )
        elif term_name == "residual":
            values = self._select_generalized_vector(dynamics_terms["residual"], env_id=env_id, side=side, robot=robot)
        elif term_name == "external":
            values = self._select_generalized_vector(
                dynamics_terms["actuation_command"]
                + dynamics_terms["contact_identification"]
                + dynamics_terms["friction"],
                env_id=env_id,
                side=side,
                robot=robot,
            )
        else:
            raise KeyError(term_name)
        return values.detach().cpu().tolist()

    def _debug_dynamics_term(
        self,
        *,
        dynamics_terms: dict[str, Any],
        term_name: str,
        env_id: int,
        side: str,
        robot,
    ) -> list[float]:
        if term_name == "inertia_leg_self":
            mass_matrix = dynamics_terms["mass_matrix"][env_id]
            joint_acc = dynamics_terms["joint_acc_for_inertia"][env_id]
            selected_mass = self._select_generalized_matrix(mass_matrix, side=side, robot=robot)
            selected_acc = self._select_generalized_vector(
                joint_acc.unsqueeze(0), env_id=0, side=side, robot=robot, allow_joint_only=True
            )
            values = (selected_mass @ selected_acc.unsqueeze(-1)).squeeze(-1)
        elif term_name == "inertia_other_joints":
            values = torch.zeros(
                self.num_dofs,
                dtype=dynamics_terms["inertia"][env_id].dtype,
                device=dynamics_terms["inertia"][env_id].device,
            )
        else:
            values = self._select_generalized_vector(dynamics_terms[term_name], env_id=env_id, side=side, robot=robot)
        return values.detach().cpu().tolist()

    def _dynamics_quality_scalars(
        self,
        *,
        dynamics_terms: dict[str, Any],
        env_id: int,
        side: str,
        robot,
    ) -> dict[str, float]:
        def selected_norm(term_name: str) -> float:
            values = self._select_generalized_vector(
                dynamics_terms[term_name], env_id=env_id, side=side, robot=robot
            ).detach()
            return float((values * values).sum().sqrt().cpu())

        residual_norm = selected_norm("residual")
        contact_norm = selected_norm("contact_force")
        inertia_norm = selected_norm("inertia")
        actuation_norm = selected_norm("actuation_command")
        contact_friction_norm = selected_norm("contact_friction") if "contact_friction" in dynamics_terms else 0.0
        implicit_drive_norm = (
            selected_norm("implicit_drive_estimate") if "implicit_drive_estimate" in dynamics_terms else 0.0
        )
        implicit_drive_saturation_norm = (
            selected_norm("implicit_drive_saturation") if "implicit_drive_saturation" in dynamics_terms else 0.0
        )
        permanent_wrench_delta_norm = 0.0
        if "permanent_wrench_total" in dynamics_terms and "tendon" in dynamics_terms:
            delta = dynamics_terms["permanent_wrench_total"] - dynamics_terms["tendon"]
            selected_delta = self._select_generalized_vector(delta, env_id=env_id, side=side, robot=robot).detach()
            permanent_wrench_delta_norm = float((selected_delta * selected_delta).sum().sqrt().cpu())
        return {
            "quality_residual_norm": residual_norm,
            "quality_contact_norm": contact_norm,
            "quality_inertia_norm": inertia_norm,
            "quality_actuation_norm": actuation_norm,
            "quality_contact_friction_norm": contact_friction_norm,
            "quality_implicit_drive_norm": implicit_drive_norm,
            "quality_implicit_drive_saturation_norm": implicit_drive_saturation_norm,
            "quality_permanent_wrench_delta_norm": permanent_wrench_delta_norm,
        }

    def record_tendon_frame(self, *, step_index: int, sim_time: float, side: str, frame: dict[str, Any]) -> None:
        """Record one visualization/debug tendon frame as timed data."""

        if not self.cfg.record_tendons:
            return
        if not self._initialized or self._tendon_db is None:
            raise RuntimeError("Call initialize(...) before record_tendon_frame(...).")
        if self._closed:
            raise RuntimeError("Cannot record after close().")
        if sim_time < self.cfg.startup_skip_seconds:
            return
        if step_index % self.cfg.sampling_stride != 0:
            return
        selected_sides = self._selected_sides()
        if side not in selected_sides and not (FULL_ROBOT_STREAM in selected_sides and side in ("left", "right")):
            return

        payload = dict(frame)
        payload.setdefault("sim_time", float(sim_time))
        self._tendon_buffer.append((int(step_index), float(sim_time), side, json.dumps(payload, sort_keys=True)))
        self._tendon_row_count += 1
        if len(self._tendon_buffer) >= self.cfg.batch_size:
            self.flush()

    def record_step(
        self,
        *,
        step_index: int,
        sim_time: float,
        robot,
        extra_context: dict[str, Any] | None = None,
        tau_override=None,
        ddq_override=None,
        dynamics_terms: dict[str, Any] | None = None,
        skip_env_ids: set[int] | None = None,
    ) -> None:
        """Record one simulation step if it passes stride and startup filters."""

        if not self._initialized or self._db is None:
            raise RuntimeError("Call initialize(...) before record_step(...).")
        if self._closed:
            raise RuntimeError("Cannot record after close().")
        if sim_time < self.cfg.startup_skip_seconds:
            return
        if step_index % self.cfg.sampling_stride != 0:
            return

        q_all = robot.data.joint_pos
        dq_all = robot.data.joint_vel
        ddq_all = self._joint_acceleration_tensor(robot, ddq_override=ddq_override)
        tau_all = self._tau_tensor(robot, tau_override=tau_override)
        base_q_all = self._base_q_tensor(robot) if self._records_base_state() else None
        base_dq_all = self._base_velocity_tensor(robot) if self._records_base_state() else None
        base_ddq_all = (
            self._base_acceleration_tensor(robot, ddq_override=ddq_override) if self._records_base_state() else None
        )
        context = dict(extra_context or {})

        excluded_env_ids = set(skip_env_ids or ())
        for env_id in self._selected_env_ids:
            if int(env_id) in excluded_env_ids:
                continue
            for side in self._selected_sides():
                joint_indices = self._joint_indices_by_side[side]
                if not self._passes_residual_filter(
                    dynamics_terms=dynamics_terms,
                    env_id=int(env_id),
                    side=side,
                    robot=robot,
                ):
                    continue
                base_coordinate_count = self._base_coordinate_count_by_side[side]
                if base_coordinate_count:
                    if base_q_all is None or base_dq_all is None or base_ddq_all is None:
                        raise RuntimeError("Internal error: base recording tensors were not prepared.")
                    q_tensor = torch.cat((base_q_all[env_id], q_all[env_id, joint_indices]), dim=0)
                    dq_tensor = torch.cat((base_dq_all[env_id], dq_all[env_id, joint_indices]), dim=0)
                    ddq_tensor = torch.cat((base_ddq_all[env_id], ddq_all[env_id, joint_indices]), dim=0)
                    tau_tensor = self._select_generalized_vector(
                        tau_all,
                        env_id=int(env_id),
                        side=side,
                        robot=robot,
                        allow_joint_only=True,
                    )
                else:
                    q_tensor = q_all[env_id, joint_indices]
                    dq_tensor = dq_all[env_id, joint_indices]
                    ddq_tensor = ddq_all[env_id, joint_indices]
                    tau_tensor = self._select_generalized_vector(
                        tau_all,
                        env_id=int(env_id),
                        side=side,
                        robot=robot,
                        allow_joint_only=True,
                    )
                q = q_tensor.detach().cpu().tolist()
                dq = dq_tensor.detach().cpu().tolist()
                ddq = ddq_tensor.detach().cpu().tolist()
                tau = tau_tensor.detach().cpu().tolist()

                self._sim_rows_by_stream.setdefault((int(env_id), side), []).append(
                    tuple(float(value) for value in [*q, *dq, *ddq, *tau])
                )
                self._row_count += 1

        if context:
            self._context_metadata.setdefault("per_step_context_seen", sorted(context))

    def _passes_residual_filter(
        self,
        *,
        dynamics_terms: dict[str, Any] | None,
        env_id: int,
        side: str,
        robot,
    ) -> bool:
        if self.cfg.residual_filter_threshold is None:
            return True
        if dynamics_terms is None:
            raise RuntimeError("residual_filter_threshold requires dynamics_terms to be provided when recording.")
        residual = self._select_generalized_vector(
            dynamics_terms["residual"], env_id=env_id, side=side, robot=robot
        ).detach()
        norm = float((residual * residual).sum().sqrt().cpu())
        return norm <= float(self.cfg.residual_filter_threshold)

    def flush(self) -> None:
        """Flush buffered rows to SQLite."""

        if self._db is None:
            return
        if self._sim_rows_by_stream:
            placeholders = ", ".join("?" for _ in self._sim_columns)
            columns = ", ".join(_quote_identifier(name) for name in self._sim_columns)
            self._db.executemany(
                f"INSERT INTO {_quote_identifier(self.cfg.sim_table_name)} ({columns}) VALUES ({placeholders})",
                self._ordered_sim_rows(),
            )
            self._sim_rows_by_stream.clear()
        if self._tendon_db is not None and self._tendon_buffer:
            self._tendon_db.executemany(
                "INSERT INTO tendon_frames (step_index, time, side, frame_json) VALUES (?, ?, ?, ?)",
                self._tendon_buffer,
            )
            self._tendon_buffer.clear()
        if self._dynamics_db is not None and self._dynamics_rows_by_stream:
            placeholders = ", ".join("?" for _ in self._dynamics_columns)
            columns = ", ".join(_quote_identifier(name) for name in self._dynamics_columns)
            self._dynamics_db.executemany(
                f"INSERT INTO dynamics_data ({columns}) VALUES ({placeholders})",
                self._ordered_dynamics_rows(),
            )
            self._dynamics_rows_by_stream.clear()
        if self._debug_db is not None and self._debug_rows_by_stream:
            placeholders = ", ".join("?" for _ in self._debug_columns)
            columns = ", ".join(_quote_identifier(name) for name in self._debug_columns)
            self._debug_db.executemany(
                f"INSERT INTO debug_data ({columns}) VALUES ({placeholders})",
                self._ordered_debug_rows(),
            )
            self._debug_rows_by_stream.clear()
        self._db.commit()
        if self._tendon_db is not None:
            self._tendon_db.commit()
        if self._dynamics_db is not None:
            self._dynamics_db.commit()
        if self._debug_db is not None:
            self._debug_db.commit()

    def drop_recent_samples(self, env_ids: set[int], *, count: int) -> int:
        """Drop recently buffered samples for selected env ids before they are flushed."""

        if count <= 0 or not env_ids:
            return 0

        dropped = 0
        for env_id in env_ids:
            for side in self._selected_sides():
                key = (int(env_id), side)
                sim_rows = self._sim_rows_by_stream.get(key)
                if sim_rows:
                    drop_count = min(count, len(sim_rows))
                    del sim_rows[-drop_count:]
                    self._row_count -= drop_count
                    dropped += drop_count

                dynamics_rows = self._dynamics_rows_by_stream.get(key)
                if dynamics_rows:
                    drop_count = min(count, len(dynamics_rows))
                    del dynamics_rows[-drop_count:]
                    self._dynamics_row_count -= drop_count

                debug_rows = self._debug_rows_by_stream.get(key)
                if debug_rows:
                    drop_count = min(count, len(debug_rows))
                    del debug_rows[-drop_count:]
                    self._debug_row_count -= drop_count
        return dropped

    def trim_to_row_count(self, target_row_count: int) -> int:
        """Trim buffered rows deterministically from the end until row_count matches target."""

        surplus = self._row_count - int(target_row_count)
        if surplus <= 0:
            return 0

        removed = 0
        for env_id in reversed(self._selected_env_ids):
            for side in reversed(self._selected_sides()):
                if surplus <= 0:
                    return removed
                key = (int(env_id), side)
                sim_rows = self._sim_rows_by_stream.get(key)
                if not sim_rows:
                    continue
                drop_count = min(surplus, len(sim_rows))
                del sim_rows[-drop_count:]
                self._row_count -= drop_count
                removed += drop_count
                surplus -= drop_count

                dynamics_rows = self._dynamics_rows_by_stream.get(key)
                if dynamics_rows:
                    del dynamics_rows[-drop_count:]
                    self._dynamics_row_count -= drop_count

                debug_rows = self._debug_rows_by_stream.get(key)
                if debug_rows:
                    del debug_rows[-drop_count:]
                    self._debug_row_count -= drop_count
        return removed

    def close(self) -> None:
        """Flush rows, write metadata, and close the SQLite connection."""

        if self._closed:
            return
        self._regularize_sim_derivatives()
        self.flush()
        if self.cfg.record_debug_dynamics:
            self._report_dynamics_residual()
        self._write_metadata()
        self._write_viz_vars()
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._tendon_db is not None:
            self._tendon_db.close()
            self._tendon_db = None
        if self._dynamics_db is not None:
            self._dynamics_db.close()
            self._dynamics_db = None
        if self._debug_db is not None:
            self._debug_db.close()
            self._debug_db = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _regularize_sim_derivatives(self) -> None:
        if self._sim_dt is None or self._sim_dt <= 0.0:
            return
        if any(self._base_coordinate_count_by_side.get(side, 0) for side in self._selected_sides()):
            self._context_metadata["kinematics_derivative_policy"] = (
                "raw simulator base linear/angular velocities and accelerations are preserved; derivative "
                "regularization is skipped for recordings with base orientation coordinates"
            )
            return

        num_dofs = self.num_dofs
        dt = float(self._sim_dt)
        regularized_rows = 0
        for key, sim_rows in self._sim_rows_by_stream.items():
            if len(sim_rows) < 3:
                continue
            dynamics_rows = self._dynamics_rows_by_stream.get(key, ())
            if len(dynamics_rows) != len(sim_rows):
                continue

            q_rows = [row[:num_dofs] for row in sim_rows]
            tau_rows = [row[3 * num_dofs : 4 * num_dofs] for row in sim_rows]
            step_indices = [int(row[1]) for row in dynamics_rows]
            dq_rows = [list(row[num_dofs : 2 * num_dofs]) for row in sim_rows]
            ddq_rows = [list(row[2 * num_dofs : 3 * num_dofs]) for row in sim_rows]

            for index in range(1, len(sim_rows) - 1):
                if step_indices[index] - step_indices[index - 1] != 1:
                    continue
                if step_indices[index + 1] - step_indices[index] != 1:
                    continue
                dq_rows[index] = [
                    (float(q_rows[index + 1][dof]) - float(q_rows[index - 1][dof])) / (2.0 * dt)
                    for dof in range(num_dofs)
                ]

            for index in range(1, len(sim_rows) - 1):
                if step_indices[index] - step_indices[index - 1] != 1:
                    continue
                if step_indices[index + 1] - step_indices[index] != 1:
                    continue
                ddq_rows[index] = [
                    (float(dq_rows[index + 1][dof]) - float(dq_rows[index - 1][dof])) / (2.0 * dt)
                    for dof in range(num_dofs)
                ]

            self._sim_rows_by_stream[key] = [
                tuple(float(value) for value in [*q_rows[index], *dq_rows[index], *ddq_rows[index], *tau_rows[index]])
                for index in range(len(sim_rows))
            ]
            regularized_rows += len(sim_rows)

        if regularized_rows:
            self._context_metadata["kinematics_derivative_policy"] = (
                "dq and ddq are recomputed from recorded q streams across consecutive samples before export"
            )
            self._context_metadata["kinematics_derivative_regularized_rows"] = regularized_rows

    def _selected_sides(self) -> tuple[str, ...]:
        if self.cfg.side_policy == "left_only":
            return ("left",)
        if self.cfg.side_policy == "right_only":
            return ("right",)
        if self.cfg.side_policy == "both_as_samples":
            return ("left", "right")
        if self.cfg.side_policy == "full_robot":
            return (FULL_ROBOT_STREAM,)
        raise NotImplementedError(f"Unsupported side_policy: {self.cfg.side_policy!r}")

    def _records_base_state(self) -> bool:
        return bool(self.cfg.record_base_state or self.cfg.side_policy == "full_robot")

    def _base_coordinate_count(self, side: str) -> int:
        if side == FULL_ROBOT_STREAM or self._records_base_state():
            return BASE_COORDINATE_COUNT
        return 0

    def _joint_names_for_side(self, side: str) -> tuple[str, ...]:
        if self.cfg.selected_joint_names is not None:
            if len(self._selected_sides()) != 1:
                raise ValueError("selected_joint_names can only be used with one selected side.")
            return tuple(self.cfg.selected_joint_names)
        if side == FULL_ROBOT_STREAM:
            if self.cfg.joint_set == "real_leg_joints":
                return (*REAL_LEG_JOINTS["left"], *REAL_LEG_JOINTS["right"])
            if self.cfg.joint_set == "tendon_chain_5":
                return (*TENDON_CHAIN_5_JOINTS["left"], *TENDON_CHAIN_5_JOINTS["right"])
            raise ValueError(f"Unknown joint_set: {self.cfg.joint_set!r}")
        if self.cfg.joint_set == "real_leg_joints":
            return REAL_LEG_JOINTS[side]
        if self.cfg.joint_set == "tendon_chain_5":
            return TENDON_CHAIN_5_JOINTS[side]
        raise ValueError(f"Unknown joint_set: {self.cfg.joint_set!r}")

    def _body_names_for_side(self, side: str) -> tuple[str, ...]:
        if self.cfg.selected_body_names is not None:
            if len(self._selected_sides()) != 1:
                raise ValueError("selected_body_names can only be used with one selected side.")
            return tuple(self.cfg.selected_body_names)
        if self.cfg.body_set != "tendon_chain_links":
            raise ValueError(f"Unknown body_set: {self.cfg.body_set!r}")
        if side == FULL_ROBOT_STREAM:
            return (*TENDON_CHAIN_LINKS["left"], *TENDON_CHAIN_LINKS["right"])
        return TENDON_CHAIN_LINKS[side]

    def _resolve_joint_indices(self, robot) -> None:
        expected_dofs: int | None = None
        for side in self._selected_sides():
            names = self._joint_names_for_side(side)
            _validate_unique(names, f"{side} selected joint names")
            indices, found_names = robot.find_joints(list(names), preserve_order=True)
            if tuple(found_names) != names:
                raise RuntimeError(f"Could not resolve {side} joints. Requested {names}; found {tuple(found_names)}")
            base_coordinate_count = self._base_coordinate_count(side)
            dof_count = base_coordinate_count + len(names)
            if expected_dofs is None:
                expected_dofs = dof_count
            elif dof_count != expected_dofs:
                raise RuntimeError("All sides must use the same number of DOFs when stored as separate samples.")
            self._joint_indices_by_side[side] = _to_int_list(indices)
            self._joint_names_by_side[side] = names
            self._base_coordinate_count_by_side[side] = base_coordinate_count

        self._sim_columns = _sim_data_columns(expected_dofs or 0)

    def _resolve_env_ids(self, robot) -> None:
        if self.cfg.selected_env_ids is None:
            self._selected_env_ids = tuple(range(robot.num_instances))
            return

        env_ids = tuple(int(env_id) for env_id in self.cfg.selected_env_ids)
        _validate_unique(env_ids, "selected environment ids")
        invalid = [env_id for env_id in env_ids if env_id < 0 or env_id >= robot.num_instances]
        if invalid:
            raise ValueError(f"Selected environment ids out of range for {robot.num_instances} envs: {invalid}")
        self._selected_env_ids = env_ids

    def _ordered_stream_keys(self) -> list[tuple[int, str]]:
        return [(env_id, side) for env_id in self._selected_env_ids for side in self._selected_sides()]

    def _ordered_sim_rows(self) -> list[tuple[float, ...]]:
        rows: list[tuple[float, ...]] = []
        for key in self._ordered_stream_keys():
            rows.extend(self._sim_rows_by_stream.get(key, ()))
        return rows

    def _ordered_dynamics_rows(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        sample_id = self._next_dynamics_sample_id
        for key in self._ordered_stream_keys():
            for row in self._dynamics_rows_by_stream.get(key, ()):
                rows.append((sample_id, *row[1:]))
                sample_id += 1
        self._next_dynamics_sample_id = sample_id
        return rows

    def _ordered_debug_rows(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        sample_id = self._next_debug_sample_id
        for key in self._ordered_stream_keys():
            for row in self._debug_rows_by_stream.get(key, ()):
                rows.append((sample_id, *row[1:]))
                sample_id += 1
        self._next_debug_sample_id = sample_id
        return rows

    def _resolve_body_indices(self, robot) -> None:
        if not self.cfg.record_spatial_state:
            return
        for side in self._selected_sides():
            names = self._body_names_for_side(side)
            _validate_unique(names, f"{side} selected body names")
            indices, found_names = robot.find_bodies(list(names), preserve_order=True)
            if tuple(found_names) != names:
                raise RuntimeError(f"Could not resolve {side} bodies. Requested {names}; found {tuple(found_names)}")
            self._body_indices_by_side[side] = _to_int_list(indices)
            self._body_names_by_side[side] = names

    def _resolve_joint_dynamics_properties(self, robot) -> None:
        rows = []
        for side, names in self._joint_names_by_side.items():
            q_offset = self._base_coordinate_count_by_side.get(side, 0)
            for q_index, (joint_name, joint_index) in enumerate(zip(names, self._joint_indices_by_side[side])):
                rows.append(
                    {
                        "side": side,
                        "q_index": q_offset + q_index,
                        "joint_name": joint_name,
                        "isaac_joint_index": int(joint_index),
                        "static_friction_coeff": _tensor_scalar(robot.data.joint_friction_coeff, joint_index),
                        "dynamic_friction_coeff": _tensor_scalar(robot.data.joint_dynamic_friction_coeff, joint_index),
                        "viscous_friction_coeff": _tensor_scalar(robot.data.joint_viscous_friction_coeff, joint_index),
                        "armature": _tensor_scalar(robot.data.joint_armature, joint_index),
                        "stiffness": _jsonable_scalar(_tensor_scalar(robot.data.joint_stiffness, joint_index)),
                        "damping": _jsonable_scalar(_tensor_scalar(robot.data.joint_damping, joint_index)),
                        "effort_limit": _jsonable_scalar(_tensor_scalar(robot.data.joint_effort_limits, joint_index)),
                        "velocity_limit": _jsonable_scalar(_tensor_scalar(robot.data.joint_vel_limits, joint_index)),
                        "soft_position_limit_lower": _jsonable_scalar(
                            _tensor_scalar(robot.data.soft_joint_pos_limits[:, :, 0], joint_index)
                        ),
                        "soft_position_limit_upper": _jsonable_scalar(
                            _tensor_scalar(robot.data.soft_joint_pos_limits[:, :, 1], joint_index)
                        ),
                    }
                )
        self._joint_dynamics_properties_rows = rows

    def _resolve_body_dynamics_properties(self, robot) -> None:
        rows = []
        try:
            masses = robot.root_physx_view.get_masses().to(robot.device)
        except Exception:
            masses = None
        try:
            coms = robot.root_physx_view.get_coms().to(robot.device)
        except Exception:
            coms = None
        try:
            inertias = robot.root_physx_view.get_inertias().to(robot.device)
        except Exception:
            inertias = None

        if masses is None and coms is None and inertias is None:
            self._body_dynamics_properties_rows = []
            return

        for env_id in self._selected_env_ids:
            for body_index, body_name in enumerate(robot.body_names):
                row: dict[str, Any] = {
                    "env_id": int(env_id),
                    "body_name": body_name,
                    "isaac_body_index": int(body_index),
                    "mass_unit": "kg",
                    "com_frame": "body/link frame",
                    "inertia_frame": "body/link frame about body COM when provided by PhysX",
                }
                if masses is not None:
                    mass_values = masses[int(env_id), int(body_index)].detach().cpu().reshape(-1).tolist()
                    row["mass"] = (
                        _jsonable_scalar(float(mass_values[0]))
                        if len(mass_values) == 1
                        else [_jsonable_scalar(float(value)) for value in mass_values]
                    )
                if coms is not None:
                    com = coms[int(env_id), int(body_index)].detach().cpu().tolist()
                    row["com_position"] = [_jsonable_scalar(float(value)) for value in com[:3]]
                    if len(com) >= 7:
                        row["com_orientation_quat_wxyz"] = [_jsonable_scalar(float(value)) for value in com[3:7]]
                if inertias is not None:
                    inertia = inertias[int(env_id), int(body_index)].detach().cpu().reshape(-1).tolist()
                    row["inertia"] = [_jsonable_scalar(float(value)) for value in inertia]
                rows.append(row)
        self._body_dynamics_properties_rows = rows

    def _create_tables(self) -> None:
        self._db = sqlite3.connect(self.sqlite_path)
        sim_columns_sql = ", ".join(f"{_quote_identifier(name)} REAL NOT NULL" for name in self._sim_columns)
        self._db.execute(f"CREATE TABLE {_quote_identifier(self.cfg.sim_table_name)} ({sim_columns_sql})")
        self._db.commit()
        if self.cfg.record_tendons:
            self._tendon_db = sqlite3.connect(self.tendon_sqlite_path)
            self._tendon_db.execute(
                """
                CREATE TABLE tendon_frames (
                    step_index INTEGER NOT NULL,
                    time REAL NOT NULL,
                    side TEXT NOT NULL,
                    frame_json TEXT NOT NULL
                )
                """
            )
            self._tendon_db.execute("CREATE INDEX tendon_frames_side_step_idx ON tendon_frames (side, step_index)")
            self._tendon_db.commit()
        if self.cfg.record_dynamics:
            self._dynamics_columns = _training_dynamics_data_columns(self.num_dofs)
            self._dynamics_db = sqlite3.connect(self.dynamics_sqlite_path)
            columns_sql = ", ".join(_dynamics_column_sql(name) for name in self._dynamics_columns)
            self._dynamics_db.execute(f"CREATE TABLE dynamics_data ({columns_sql})")
            self._dynamics_db.execute("CREATE INDEX dynamics_data_step_idx ON dynamics_data (step_index, side)")
            self._dynamics_db.commit()
        if self.cfg.record_debug_dynamics:
            self._debug_matrix_term_names = _debug_matrix_term_names_for_num_dofs(self.num_dofs)
            self._debug_columns = _debug_dynamics_data_columns(
                self.num_dofs,
                matrix_term_names=self._debug_matrix_term_names,
            )
            self._debug_db = sqlite3.connect(self.debug_sqlite_path)
            columns_sql = ", ".join(_dynamics_column_sql(name) for name in self._debug_columns)
            self._debug_db.execute(f"CREATE TABLE debug_data ({columns_sql})")
            self._debug_db.execute("CREATE INDEX debug_data_step_idx ON debug_data (step_index, side)")
            self._debug_db.commit()

    def _tau_tensor(self, robot, *, tau_override=None):
        if tau_override is not None:
            return tau_override
        if self.cfg.tau_source == "actuation_command":
            return actuation_command_tensor(robot)
        if self.cfg.tau_source == "motor_torque":
            return motor_torque_tensor(robot)
        if self.cfg.tau_source == "controller_plus_ground":
            raise RuntimeError("tau_source='controller_plus_ground' requires a tau_override tensor.")
        if self.cfg.tau_source == "applied_torque":
            return robot.data.applied_torque
        if self.cfg.tau_source == "computed_torque":
            return robot.data.computed_torque
        if self.cfg.tau_source == "zero":
            return robot.data.joint_pos * 0.0
        raise ValueError(f"Unsupported tau_source: {self.cfg.tau_source!r}")

    def _base_q_tensor(self, robot) -> torch.Tensor:
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        return torch.cat(
            (
                robot.data.root_pos_w,
                torch.stack((roll, pitch, yaw), dim=-1),
            ),
            dim=-1,
        )

    def _base_velocity_tensor(self, robot) -> torch.Tensor:
        return robot.data.root_com_vel_w

    def _base_acceleration_tensor(self, robot, *, ddq_override=None) -> torch.Tensor:
        if self.cfg.ddq_source == "physx_raw":
            return self._raw_base_acceleration_tensor(robot)
        if self.cfg.ddq_source == "recording_interval":
            if isinstance(ddq_override, dict) and "root_acc" in ddq_override:
                return ddq_override["root_acc"]
            return self._raw_base_acceleration_tensor(robot)
        raise ValueError(f"Unsupported ddq_source: {self.cfg.ddq_source!r}")

    def _raw_base_acceleration_tensor(self, robot) -> torch.Tensor:
        if robot.is_fixed_base:
            return torch.zeros(
                (robot.num_instances, BASE_COORDINATE_COUNT),
                dtype=robot.data.joint_acc.dtype,
                device=robot.device,
            )
        try:
            return robot.root_physx_view.get_link_accelerations()[:, 0, :]
        except Exception:
            pass
        root_acc = getattr(robot.data, "root_com_acc_w", None)
        if root_acc is not None:
            return root_acc
        return robot.data.body_com_acc_w[:, 0, :]

    def _joint_acceleration_tensor(self, robot, *, ddq_override=None) -> torch.Tensor:
        if self.cfg.ddq_source == "physx_raw":
            return robot.data.joint_acc
        if self.cfg.ddq_source == "recording_interval":
            if isinstance(ddq_override, dict):
                joint_acc = ddq_override.get("joint_acc")
                if joint_acc is not None:
                    return joint_acc
            if ddq_override is not None and not isinstance(ddq_override, dict):
                return ddq_override
            return robot.data.joint_acc
        raise ValueError(f"Unsupported ddq_source: {self.cfg.ddq_source!r}")

    def _select_generalized_vector(
        self,
        values: torch.Tensor,
        *,
        env_id: int,
        side: str,
        robot,
        allow_joint_only: bool = False,
    ) -> torch.Tensor:
        """Select one recorded generalized-coordinate vector from a joint or full generalized tensor."""

        row = values[env_id]
        joint_count = int(robot.data.joint_pos.shape[1])
        base_count = self._base_coordinate_count_by_side[side]
        joint_indices = self._joint_indices_by_side[side]
        if row.shape[-1] == joint_count:
            if base_count and not allow_joint_only:
                raise RuntimeError(
                    "Full robot recording requires dynamics terms with base generalized-force slots. "
                    f"Got joint-only tensor with {joint_count} entries."
                )
            selected = row[joint_indices]
            if base_count:
                return torch.cat((row.new_zeros(base_count), selected), dim=0)
            return selected

        full_count = BASE_COORDINATE_COUNT + joint_count
        if row.shape[-1] == full_count:
            full_indices = [*range(base_count), *[BASE_COORDINATE_COUNT + int(index) for index in joint_indices]]
            return row[full_indices]

        raise RuntimeError(
            f"Cannot select {side!r} generalized coordinates from tensor width {row.shape[-1]}; "
            f"expected {joint_count} or {full_count}."
        )

    def _select_generalized_matrix(self, matrix: torch.Tensor, *, side: str, robot) -> torch.Tensor:
        joint_count = int(robot.data.joint_pos.shape[1])
        base_count = self._base_coordinate_count_by_side[side]
        joint_indices = self._joint_indices_by_side[side]
        if matrix.shape[-1] == joint_count and matrix.shape[-2] == joint_count:
            selected_joint = matrix[joint_indices][:, joint_indices]
            if not base_count:
                return selected_joint
            selected = matrix.new_zeros((base_count + len(joint_indices), base_count + len(joint_indices)))
            selected[base_count:, base_count:] = selected_joint
            return selected

        full_count = BASE_COORDINATE_COUNT + joint_count
        if matrix.shape[-1] == full_count and matrix.shape[-2] == full_count:
            full_indices = [*range(base_count), *[BASE_COORDINATE_COUNT + int(index) for index in joint_indices]]
            return matrix[full_indices][:, full_indices]

        raise RuntimeError(
            f"Cannot select {side!r} generalized-coordinate matrix from shape {tuple(matrix.shape)}; "
            f"expected ({joint_count}, {joint_count}) or ({full_count}, {full_count})."
        )

    def _write_metadata(self) -> None:
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sqlite_path": str(self.sqlite_path),
            "tendon_sqlite_path": str(self.tendon_sqlite_path) if self.cfg.record_tendons else None,
            "dynamics_sqlite_path": str(self.dynamics_sqlite_path) if self.cfg.record_dynamics else None,
            "debug_sqlite_path": str(self.debug_sqlite_path) if self.cfg.record_debug_dynamics else None,
            "viz_vars_path": str(self.viz_vars_path),
            "sim_table_name": self.cfg.sim_table_name,
            "dynamics_table_name": "dynamics_data" if self.cfg.record_dynamics else None,
            "debug_table_name": "debug_data" if self.cfg.record_debug_dynamics else None,
            "num_dofs": self.num_dofs,
            "row_count": self._row_count,
            "tendon_row_count": self._tendon_row_count,
            "dynamics_row_count": self._dynamics_row_count,
            "debug_row_count": self._debug_row_count,
            "sim_columns": self._sim_columns,
            "dynamics_columns": self._dynamics_columns,
            "debug_columns": self._debug_columns,
            "debug_matrix_terms": list(self._debug_matrix_term_names) if self.cfg.record_debug_dynamics else [],
            "sim_dt": self._sim_dt,
            "selected_env_ids": list(self._selected_env_ids),
            "sample_order": "selected_env_ids in order, then selected_sides in order, then all recorded steps",
            "tau_source": self.cfg.tau_source,
            "ddq_source": self.cfg.ddq_source,
            "residual_filter_threshold": self.cfg.residual_filter_threshold,
            "tau_semantics": self._tau_semantics(),
            "ddq_semantics": self._ddq_semantics(),
            "sim_column_semantics": self._sim_column_semantics(),
            "dynamics_semantics": self._dynamics_semantics(),
            "force_frame_conventions": self._force_frame_conventions(),
            "base_constraint_policy": self._base_constraint_policy(),
            "sysid_model_policy": self._sysid_model_policy(),
            "available_tau_sources": [
                "actuation_command",
                "motor_torque",
                "controller_plus_ground",
                "applied_torque",
                "computed_torque",
                "zero",
            ],
            "available_ddq_sources": [
                "physx_raw",
                "recording_interval",
            ],
            "sim_units": self._sim_units(),
            "dynamics_units": {"tau": "per-coordinate generalized force; see coordinate_mappings[].tau_unit"},
            "coordinate_mappings": self._coordinate_metadata(),
            "joint_mappings": self._joint_metadata(),
            "joint_dynamics_properties": self._joint_dynamics_properties(),
            "body_mappings": self._body_metadata(),
            "body_dynamics_properties": self._body_dynamics_properties(),
            "omitted_joint_names": self._omitted_joint_metadata(),
            "config": _jsonable_config(self.cfg),
            "runtime_metadata": self._context_metadata,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_viz_vars(self) -> None:
        viz_vars = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format_version": 1,
            "recording_dir": str(self.output_dir),
            "kinematics_db": self.cfg.sqlite_filename,
            "tendons_db": self.cfg.tendon_sqlite_filename if self.cfg.record_tendons else None,
            "dynamics_db": self.cfg.dynamics_sqlite_filename if self.cfg.record_dynamics else None,
            "debug_db": self.cfg.debug_sqlite_filename if self.cfg.record_debug_dynamics else None,
            "metadata": self.cfg.metadata_filename,
            "sim_table_name": self.cfg.sim_table_name,
            "tendon_table_name": "tendon_frames",
            "dynamics_table_name": "dynamics_data",
            "debug_table_name": "debug_data" if self.cfg.record_debug_dynamics else None,
            "tendon_frame_format": "jsonl-compatible analytic tendon debug frame",
            "num_dofs": self.num_dofs,
            "sim_dt": self._sim_dt,
            "ddq_source": self.cfg.ddq_source,
            "ddq_semantics": self._ddq_semantics(),
            "base_constraint_policy": self._base_constraint_policy(),
            "sysid_model_policy": self._sysid_model_policy(),
            "selected_sides": list(self._selected_sides()),
            "selected_env_ids": list(self._selected_env_ids),
            "sample_order": "selected_env_ids in order, then selected_sides in order, then all recorded steps",
            "coordinate_mappings": self._coordinate_metadata(),
            "joint_mappings": self._joint_metadata(),
            "body_mappings": self._body_metadata(),
            "config": _jsonable_config(self.cfg),
        }
        self.viz_vars_path.write_text(json.dumps(viz_vars, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _omitted_joint_metadata(self) -> dict[str, list[str]]:
        omitted_by_side = OMITTED_JOINTS_BY_SET.get(self.cfg.joint_set)
        if omitted_by_side is None:
            return {side: [] for side in self._selected_sides()}
        result = {}
        for side in self._selected_sides():
            if side == FULL_ROBOT_STREAM:
                result[side] = [*omitted_by_side["left"], *omitted_by_side["right"]]
            else:
                result[side] = list(omitted_by_side[side])
        return result

    def _tau_semantics(self) -> str:
        if self.cfg.tau_source == "actuation_command":
            return (
                "Identix actuation-only generalized torque label used by the current dynamics balance: "
                "Forrest motor torques with the knee-flexor coordinates removed. Contact, friction, tendon, "
                "gravity, implicit-drive diagnostics, solver, and constraint forces are intentionally excluded "
                "and recorded separately in dynamics/debug tables."
            )
        if self.cfg.tau_source == "motor_torque":
            return (
                "motor actuation generalized torque only: robot.data.applied_torque is zeroed on every joint except "
                "the Forrest motor joints listed in isaaclab.tendons.models.analytic.constants.actuated_joint_names; "
                "contact, tendon, passive damping, solver, and constraint forces are intentionally excluded"
            )
        if self.cfg.tau_source == "controller_plus_ground":
            return (
                "legacy mixed label: actuator generalized torque on actuated joints plus PhysX ground contact "
                "generalized torque from measured contact wrench projected with J^T; uses normal plus friction "
                "contact force and contact-point moment when the contact sensor provides them; tendon forces are "
                "intentionally excluded"
            )
        if self.cfg.tau_source == "zero":
            return "zero placeholder for Identix kinematics schema compatibility; do not use as dynamics labels"
        if self.cfg.tau_source == "applied_torque":
            return "IsaacLab robot.data.applied_torque selected by joint index; may not include tendon body wrenches"
        if self.cfg.tau_source == "computed_torque":
            return "IsaacLab robot.data.computed_torque selected by joint index"
        return self.cfg.tau_source

    def _ddq_semantics(self) -> str:
        if self.cfg.ddq_source == "physx_raw":
            return (
                "raw PhysX generalized acceleration sampled from robot.data.joint_acc and the root link "
                "acceleration from root_physx_view.get_link_accelerations()[:, 0, :]. This is the acceleration "
                "used for sim_data ddq and for tau_inertia = M(q) * ddq in the recorded dynamics balance."
            )
        if self.cfg.ddq_source == "recording_interval":
            return (
                "adjacent-recording-interval finite difference of dq. This is useful as a kinematic diagnostic "
                "but is not the default sysid source because the latest residual audit matched raw PhysX "
                "accelerations better."
            )
        return self.cfg.ddq_source

    def _dynamics_semantics(self) -> dict[str, str]:
        if not self.cfg.record_dynamics:
            return {}
        return {
            "sample_id": "zero-based row index aligned one-to-one with sim_data rowid - 1 after optional filtering",
            "equation": (
                "Identix convention: residual = conservative - non_conservative, where conservative = "
                "tau_inertia + tau_coriolis + tau_gravity + tau_tendon and non_conservative = "
                "tau_actuation + tau_contact + tau_friction"
            ),
            "tau_inertia": (
                "selected generalized rows of the full floating-base generalized inertia product M(q)qdd, using "
                "the same acceleration source stored in sim_data ddq"
            ),
            "tau_coriolis": (
                "Coriolis/centrifugal term on the left-hand side of the training equation. In RL recordings "
                "this is the joint-coordinate balance term from PhysX direct generalized-force output when "
                "available, because that source matched the complete simulator force balance best. The "
                "compensation-derived term is retained as a debug/fallback convention check."
            ),
            "tau_gravity": (
                "sysid gravity term on the left-hand side of the training equation: PhysX direct generalized "
                "joint gravity when available, corrected with the floating-base gravity wrench inferred from "
                "gravity_compensation_actual. Compensation-derived gravity is retained as a debug/fallback "
                "convention check."
            ),
            "tau_tendon": "analytic tendon term on the left-hand side of the training equation",
            "tau_actuation": (
                "external actuation generalized force used by the dynamics force balance; contains motor torque "
                "except pantograph implicit effort and knee-flexor effort, which are kept as debug diagnostics"
            ),
            "tau_contact": (
                "measured contact generalized force used by the identification balance, using contact_validated. "
                "When recording detail sensors are available, contact is the sum of normal and tangential/friction "
                "contact forces plus their contact-point lever-arm moments. The lever-arm moment is projected with "
                "force x (contact_pos_w - body_pos_w), matching the contact sensor force sign to the angular "
                "Jacobian wrench convention. Contact terms are world-frame wrenches projected with J^T."
            ),
            "tau_contact_normal": (
                "debug-only measured normal contact contribution, including normal-force lever-arm moment when "
                "contact points are available"
            ),
            "tau_contact_friction": (
                "debug-only measured tangential/friction contact contribution, including friction-force lever-arm "
                "moment when contact points are available"
            ),
            "tau_implicit_drive_estimate": (
                "debug-only estimate of implicit PhysX drive generalized force: stiffness*(q_target-q) + "
                "damping*(dq_target-dq) + effort_target, clipped by joint effort limits"
            ),
            "tau_permanent_wrench_total": (
                "debug-only projection of the articulation permanent body-wrench composer, expressed in the same "
                "generalized-force convention as tau_tendon; permanent_wrench_total - tendon should be near zero "
                "if tendon is the only persistent body-wrench source"
            ),
            "tau_friction": "external configured joint-friction model term",
            "tau_solver_constraint_internal": (
                "optional PhysX solver-projected passive/limit constraint reaction generalized force for selected "
                "passive chain, pulley, metatarsophalangeal, and knee-flexor joints. It is recorded for sysid "
                "diagnostics/post-processing, but is not included in the default Identix residual equation unless "
                "the consumer explicitly chooses a constraint-reaction model."
            ),
            "tau_residual": (
                "default force-balance residual vector: tau_inertia + tau_gravity + tau_coriolis + tau_tendon - "
                "(tau_actuation + tau_contact + tau_friction). This is a quality/weighting signal, not an "
                "additional physical input force."
            ),
            "tau_external": "tau_actuation + tau_contact + tau_friction",
            "coordinate_order": (
                "when base coordinates are enabled, q0..q5 are base x/y/z/roll/pitch/yaw and tau0..tau5 are "
                "world-frame force x/y/z plus torque x/y/z; remaining coordinates are selected Isaac joints"
            ),
        }

    def _sim_column_semantics(self) -> dict[str, Any]:
        return {
            "layout": (
                "sim_data columns are [q0..qN-1, dq0..dqN-1, ddq0..ddqN-1, tau0..tauN-1]. "
                "Use coordinate_mappings[].q_index to map each i to the floating-base coordinate or Isaac joint."
            ),
            "q": "position coordinates; base orientation is roll/pitch/yaw from Isaac root quaternion when present",
            "dq": (
                "coordinate velocities; base linear/angular entries are raw Isaac root COM linear velocity and "
                "world angular velocity, not Euler-angle rates"
            ),
            "ddq": "coordinate accelerations using ddq_source; physx_raw uses raw PhysX root/joint accelerations",
            "tau": self._tau_semantics(),
        }

    def _force_frame_conventions(self) -> dict[str, str]:
        return {
            "base_generalized_force": (
                "tau0..tau2 are world-frame force components [Fx,Fy,Fz] in N. tau3..tau5 are world-frame "
                "torque components [Tx,Ty,Tz] in N*m."
            ),
            "joint_generalized_force": (
                "joint tau entries are N*m in the positive Isaac joint coordinate convention listed in "
                "coordinate_mappings."
            ),
            "contact": (
                "contact sensor forces are world-frame wrenches projected with J^T. Detail sensors split normal "
                "and tangential/friction contact. Contact-point moments use force x (contact_pos_w - body_pos_w) "
                "to match the established recorder sign convention."
            ),
            "tendon_and_permanent_wrench": (
                "tendon and permanent-wrench diagnostics originate as body/link-frame wrenches, are rotated to "
                "world frame, projected with J^T, and stored with the sign convention used by tau_tendon."
            ),
            "gravity": (
                "gravity_identification is the exported gravity term. external_base_gravity and compensation "
                "variants are debug terms for checking floating-base convention only."
            ),
        }

    def _base_constraint_policy(self) -> dict[str, Any]:
        mode = str(self.cfg.constraint_mode or "unknown")
        task_name = str(self._context_metadata.get("task", ""))
        task_lower = task_name.lower()
        mode_lower = mode.lower()
        full_base_recorded = any(
            self._base_coordinate_count_by_side.get(side, 0) >= BASE_COORDINATE_COUNT for side in self._selected_sides()
        )
        boom_detected = "boom" in mode_lower or "boom" in task_lower
        static_detected = "static" in mode_lower
        unmodeled_base_constraint = full_base_recorded and (boom_detected or static_detected)
        if unmodeled_base_constraint:
            guidance = (
                "Do not use these full-base samples as free-floating robot sysid data unless the boom/static "
                "reaction wrench is measured separately. Use a non-Boom RL task or run.py --constraint_mode freefall "
                "for full-base sysid; leg-only recordings can still be useful if the base motion is treated as "
                "prescribed rather than identified."
            )
        else:
            guidance = (
                "No configured world-to-base boom/static constraint was detected for the recorded base coordinates."
            )
        return {
            "constraint_mode": mode,
            "runtime_task": task_name or None,
            "full_base_coordinates_recorded": bool(full_base_recorded),
            "boom_constraint_detected": bool(boom_detected),
            "static_base_constraint_detected": bool(static_detected),
            "unmodeled_external_base_constraint": bool(unmodeled_base_constraint),
            "measured_reaction_wrench": (
                "not recorded; the current recorder covers actuation, measured contact, friction, gravity, Coriolis, "
                "tendon/permanent body wrenches, and PhysX articulation projected joint forces, but not external "
                "world-to-base D6/fixed-joint reactions"
            ),
            "sysid_guidance": guidance,
        }

    def _maybe_print_base_constraint_warning(self) -> None:
        policy = self._base_constraint_policy()
        if not policy["unmodeled_external_base_constraint"]:
            return
        print(
            "[WARN] Forrest Identix recording includes full base coordinates while a "
            f"{policy['constraint_mode']!r} base constraint/task is active. The boom/static reaction wrench is not "
            "recorded; use a freefall/non-Boom recording for free-floating full-base sysid."
        )

    def _sysid_model_policy(self) -> dict[str, Any]:
        return {
            "selected_model": "current",
            "selected_equation": (
                "tau_inertia + tau_gravity_identification + tau_coriolis + tau_tendon = "
                "tau_actuation_command + tau_contact_identification + tau_friction + residual"
            ),
            "selected_ddq_source": self.cfg.ddq_source,
            "selected_tau_source": self.cfg.tau_source,
            "base_acceleration_convention": (
                "world-frame root link acceleration ordered as [linear_x, linear_y, linear_z, "
                "angular_x, angular_y, angular_z]"
            ),
            "gravity_policy": (
                "use the recorded gravity_identification term: PhysX direct generalized joint gravity when "
                "available, with the floating-base gravity wrench corrected from gravity_compensation_actual. "
                "The full external compensation-gravity candidate is retained as a debug comparison, but is not "
                "the export convention because it can cancel/move joint gravity terms across sides."
            ),
            "coriolis_policy": (
                "use the recorded coriolis term: PhysX direct generalized Coriolis/centrifugal force when "
                "available. This source is retained because the residual audits showed it is the physical "
                "simulator-consistent convention; compensation-derived Coriolis remains recorded only as a "
                "debug/fallback convention check."
            ),
            "contact_policy": (
                "use measured contact_identification, currently equal to contact_validated: normal and friction "
                "contact plus accepted contact-point moment projected with J^T in world-frame wrench convention"
            ),
            "actuation_policy": (
                "sim_data tau and tau_actuation are actuation_command only; contact, gravity, Coriolis, tendon, "
                "friction, implicit-drive diagnostics, solver, and constraint forces remain separate dynamics/debug "
                "terms"
            ),
            "base_constraint_policy": self._base_constraint_policy(),
            "debug_candidate_policy": (
                "lower residual debug candidates are not adopted unless they preserve the complete physical "
                "force balance and the same convention is exported in sim_data and metadata"
            ),
        }

    def _report_dynamics_residual(self) -> None:  # noqa: C901
        if not self.cfg.record_debug_dynamics or self._debug_db is None or self._debug_row_count == 0:
            return

        candidate_specs = self._debug_residual_candidate_specs()
        diagnostic_terms = (
            "residual",
            "inertia",
            "inertia_recording_interval",
            "inertia_raw",
            "inertia_physx_base_recording_joints",
            "inertia_recording_base_physx_joints",
            "inertia_physx_base_body_frame",
            "inertia_physx_base_swapped",
            "inertia_joint_only",
            "inertia_root_coupling",
            "coriolis",
            "coriolis_force_api",
            "coriolis_compensation_actual",
            "coriolis_api_delta",
            "gravity",
            "gravity_identification",
            "gravity_force_api",
            "gravity_compensation_actual",
            "gravity_api_delta",
            "external_base_gravity",
            "tendon",
            "tendon_model",
            "tendon_projection_delta",
            "actuation_command",
            "motor_actuation",
            "physx_actuation",
            "pantograph_reconstructed_actuation",
            "pantograph_actuation_error",
            "contact",
            "contact_force",
            "contact_moment",
            "contact_normal",
            "contact_friction",
            "contact_validated",
            "contact_identification",
            "contact_digit",
            "contact_connector",
            "contact_base",
            "contact_self_collision",
            "friction",
            "implicit_drive_estimate",
            "permanent_wrench_total",
            "solver_joint",
            "solver_constraint_internal",
        )
        selected_terms = sorted(
            {
                _weighted_term_name(weighted_term)
                for spec in candidate_specs
                for weighted_term in (*spec["conservative_terms"], *spec["non_conservative_terms"])
            }
            | set(diagnostic_terms)
        )
        available_terms = [
            term
            for term in selected_terms
            if all(f"tau_{term}{i}" in self._debug_columns for i in range(self.num_dofs))
        ]
        selected_scalar_columns = [name for name in DEBUG_DYNAMICS_SCALAR_NAMES if name in self._debug_columns]
        selected_cols = (
            ["sample_id", "step_index", "env_id", "side"]
            + [f"tau_{term}{i}" for term in available_terms for i in range(self.num_dofs)]
            + selected_scalar_columns
        )
        col_index = {name: index for index, name in enumerate(selected_cols)}
        quoted_cols = ", ".join(_quote_identifier(name) for name in selected_cols)
        rows = self._debug_db.execute(f"SELECT {quoted_cols} FROM debug_data ORDER BY sample_id").fetchall()
        if not rows:
            return

        available_term_set = set(available_terms)
        active_specs = [
            spec
            for spec in candidate_specs
            if all(
                _weighted_term_name(weighted_term) in available_term_set
                for weighted_term in (*spec["conservative_terms"], *spec["non_conservative_terms"])
            )
        ]
        if not active_specs:
            return

        sample_quality, quality_pair_summary = self._debug_sample_kinematic_quality(rows, col_index)
        clean_sample_ids, dirty_sample_ids, quality_summary = self._debug_quality_subsets(
            sample_quality,
            quality_pair_summary=quality_pair_summary,
        )
        coordinate_groups = self._coordinate_report_groups()
        candidate_norms: dict[str, list[float]] = {spec["label"]: [] for spec in active_specs}
        candidate_clean_norms: dict[str, list[float]] = {spec["label"]: [] for spec in active_specs}
        candidate_dirty_norms: dict[str, list[float]] = {spec["label"]: [] for spec in active_specs}
        candidate_group_norms: dict[str, dict[str, list[float]]] = {
            spec["label"]: {group_name: [] for group_name in self._all_report_group_names(coordinate_groups)}
            for spec in active_specs
        }
        candidate_meta: dict[str, dict[str, Any]] = {spec["label"]: spec for spec in active_specs}
        candidate_worst: dict[str, tuple[float, tuple[Any, ...], tuple[float, ...]]] = {}
        component_norms: dict[str, list[float]] = {term: [] for term in diagnostic_terms if term in available_term_set}
        scalar_norms: dict[str, list[float]] = {name: [] for name in selected_scalar_columns}
        delta_norms: dict[str, list[float]] = {
            "stored_residual_minus_recomputed_baseline": [],
            "inertia_minus_recording_interval": [],
            "inertia_minus_raw": [],
            "inertia_raw_minus_physx_base_recording_joints": [],
            "inertia_raw_minus_recording_base_physx_joints": [],
            "inertia_raw_minus_physx_base_body_frame": [],
            "inertia_raw_minus_physx_base_swapped": [],
            "inertia_minus_joint_only": [],
            "coriolis_force_api_minus_compensation": [],
            "gravity_force_api_minus_compensation": [],
            "tendon_projection_minus_model": [],
            "contact_minus_validated": [],
            "contact_identification_minus_validated": [],
            "contact_minus_group_sum": [],
            "actuation_command_minus_motor": [],
            "actuation_command_minus_physx": [],
            "actuation_command_minus_implicit_drive": [],
            "permanent_wrench_total_minus_tendon": [],
            "pantograph_reconstruction_error": [],
        }
        contact_force_norms: list[float] = []
        contact_moment_norms: list[float] = []
        contact_moment_reject_count = 0
        nonzero_contact_count = 0

        for row in rows:
            sample_id = int(row[col_index["sample_id"]])
            side = str(row[col_index["side"]])
            terms = {
                term: tuple(float(row[col_index[f"tau_{term}{i}"]]) for i in range(self.num_dofs))
                for term in available_terms
            }
            for term in component_norms:
                component_norms[term].append(_vector_norm(terms[term]))
            for name in scalar_norms:
                scalar_norms[name].append(float(row[col_index[name]]))

            for spec in active_specs:
                coordinate_masks = coordinate_groups.get(side, {})
                residual = _vector_sub(
                    _weighted_vector_sum(
                        terms,
                        spec["conservative_terms"],
                        self.num_dofs,
                        coordinate_masks=coordinate_masks,
                    ),
                    _weighted_vector_sum(
                        terms,
                        spec["non_conservative_terms"],
                        self.num_dofs,
                        coordinate_masks=coordinate_masks,
                    ),
                )
                residual_norm = _vector_norm(residual)
                label = spec["label"]
                candidate_norms[label].append(residual_norm)
                if sample_id in clean_sample_ids:
                    candidate_clean_norms[label].append(residual_norm)
                if sample_id in dirty_sample_ids:
                    candidate_dirty_norms[label].append(residual_norm)
                for group_name, indices in coordinate_groups.get(side, {}).items():
                    candidate_group_norms[label][group_name].append(_vector_norm(residual, indices))
                if label not in candidate_worst or residual_norm > candidate_worst[label][0]:
                    candidate_worst[label] = (residual_norm, row, residual)

            if "residual" in terms and "current" in candidate_norms:
                coordinate_masks = coordinate_groups.get(side, {})
                recomputed = _vector_sub(
                    _weighted_vector_sum(
                        terms,
                        candidate_meta["current"]["conservative_terms"],
                        self.num_dofs,
                        coordinate_masks=coordinate_masks,
                    ),
                    _weighted_vector_sum(
                        terms,
                        candidate_meta["current"]["non_conservative_terms"],
                        self.num_dofs,
                        coordinate_masks=coordinate_masks,
                    ),
                )
                delta_norms["stored_residual_minus_recomputed_baseline"].append(
                    _vector_norm(_vector_sub(terms["residual"], recomputed))
                )
            if "inertia" in terms and "inertia_recording_interval" in terms:
                delta_norms["inertia_minus_recording_interval"].append(
                    _vector_norm(_vector_sub(terms["inertia"], terms["inertia_recording_interval"]))
                )
            if "inertia" in terms and "inertia_raw" in terms:
                delta_norms["inertia_minus_raw"].append(
                    _vector_norm(_vector_sub(terms["inertia"], terms["inertia_raw"]))
                )
            if "inertia_raw" in terms and "inertia_physx_base_recording_joints" in terms:
                delta_norms["inertia_raw_minus_physx_base_recording_joints"].append(
                    _vector_norm(_vector_sub(terms["inertia_raw"], terms["inertia_physx_base_recording_joints"]))
                )
            if "inertia_raw" in terms and "inertia_recording_base_physx_joints" in terms:
                delta_norms["inertia_raw_minus_recording_base_physx_joints"].append(
                    _vector_norm(_vector_sub(terms["inertia_raw"], terms["inertia_recording_base_physx_joints"]))
                )
            if "inertia_raw" in terms and "inertia_physx_base_body_frame" in terms:
                delta_norms["inertia_raw_minus_physx_base_body_frame"].append(
                    _vector_norm(_vector_sub(terms["inertia_raw"], terms["inertia_physx_base_body_frame"]))
                )
            if "inertia_raw" in terms and "inertia_physx_base_swapped" in terms:
                delta_norms["inertia_raw_minus_physx_base_swapped"].append(
                    _vector_norm(_vector_sub(terms["inertia_raw"], terms["inertia_physx_base_swapped"]))
                )
            if "inertia" in terms and "inertia_joint_only" in terms:
                delta_norms["inertia_minus_joint_only"].append(
                    _vector_norm(_vector_sub(terms["inertia"], terms["inertia_joint_only"]))
                )
            if "coriolis_api_delta" in terms:
                delta_norms["coriolis_force_api_minus_compensation"].append(_vector_norm(terms["coriolis_api_delta"]))
            if "gravity_api_delta" in terms:
                delta_norms["gravity_force_api_minus_compensation"].append(_vector_norm(terms["gravity_api_delta"]))
            if "tendon_projection_delta" in terms:
                delta_norms["tendon_projection_minus_model"].append(_vector_norm(terms["tendon_projection_delta"]))
            if "contact" in terms and "contact_validated" in terms:
                delta_norms["contact_minus_validated"].append(
                    _vector_norm(_vector_sub(terms["contact"], terms["contact_validated"]))
                )
            if "contact_identification" in terms and "contact_validated" in terms:
                delta_norms["contact_identification_minus_validated"].append(
                    _vector_norm(_vector_sub(terms["contact_identification"], terms["contact_validated"]))
                )
            contact_group_terms = [
                f"contact_{group_name}" for group_name in CONTACT_GROUP_NAMES if f"contact_{group_name}" in terms
            ]
            if "contact" in terms and contact_group_terms:
                contact_group_sum = _vector_add(*(terms[term] for term in contact_group_terms))
                delta_norms["contact_minus_group_sum"].append(
                    _vector_norm(_vector_sub(terms["contact"], contact_group_sum))
                )
            if "actuation_command" in terms and "motor_actuation" in terms:
                delta_norms["actuation_command_minus_motor"].append(
                    _vector_norm(_vector_sub(terms["actuation_command"], terms["motor_actuation"]))
                )
            if "actuation_command" in terms and "physx_actuation" in terms:
                delta_norms["actuation_command_minus_physx"].append(
                    _vector_norm(_vector_sub(terms["actuation_command"], terms["physx_actuation"]))
                )
            if "actuation_command" in terms and "implicit_drive_estimate" in terms:
                delta_norms["actuation_command_minus_implicit_drive"].append(
                    _vector_norm(_vector_sub(terms["actuation_command"], terms["implicit_drive_estimate"]))
                )
            if "permanent_wrench_total" in terms and "tendon" in terms:
                delta_norms["permanent_wrench_total_minus_tendon"].append(
                    _vector_norm(_vector_sub(terms["permanent_wrench_total"], terms["tendon"]))
                )
            if "pantograph_actuation_error" in terms:
                delta_norms["pantograph_reconstruction_error"].append(_vector_norm(terms["pantograph_actuation_error"]))
            if "contact_force" in terms and "contact_moment" in terms:
                contact_force_norm = _vector_norm(terms["contact_force"])
                contact_moment_norm = _vector_norm(terms["contact_moment"])
                contact_force_norms.append(contact_force_norm)
                contact_moment_norms.append(contact_moment_norm)
                if contact_force_norm > 1.0e-6:
                    nonzero_contact_count += 1
                if contact_moment_norm > 2.0 * max(contact_force_norm, 1.0e-6):
                    contact_moment_reject_count += 1

        candidate_summaries = self._debug_candidate_summaries(
            candidate_norms,
            candidate_meta=candidate_meta,
            candidate_group_norms=candidate_group_norms,
        )
        clean_candidate_summaries = self._debug_candidate_summaries(
            candidate_clean_norms,
            candidate_meta=candidate_meta,
        )
        dirty_candidate_summaries = self._debug_candidate_summaries(
            candidate_dirty_norms,
            candidate_meta=candidate_meta,
        )
        model_candidate_summaries = self._summaries_for_role(candidate_summaries, "model_candidate")
        diagnostic_control_summaries = self._summaries_for_role(candidate_summaries, "diagnostic_control")
        clean_model_candidate_summaries = self._summaries_for_role(clean_candidate_summaries, "model_candidate")
        clean_diagnostic_control_summaries = self._summaries_for_role(clean_candidate_summaries, "diagnostic_control")
        dirty_model_candidate_summaries = self._summaries_for_role(dirty_candidate_summaries, "model_candidate")
        dirty_diagnostic_control_summaries = self._summaries_for_role(dirty_candidate_summaries, "diagnostic_control")
        best_by_family = self._best_by_family(candidate_summaries)
        best_model_by_family = self._best_by_family(model_candidate_summaries)
        best_diagnostic_by_family = self._best_by_family(diagnostic_control_summaries)
        current_residual_thresholds = _threshold_percentages(candidate_norms.get("current", []))
        best_model_label = model_candidate_summaries[0]["label"] if model_candidate_summaries else None
        best_model_residual_thresholds = (
            _threshold_percentages(candidate_norms.get(best_model_label, [])) if best_model_label else []
        )

        diagnostics = (
            {name: _norm_summary(values) for name, values in delta_norms.items() if values}
            | {f"{name}_norm": _norm_summary(values) for name, values in component_norms.items() if values}
            | {name: _norm_summary(values) for name, values in scalar_norms.items() if values}
        )
        if contact_force_norms:
            diagnostics["contact_moment_validation"] = {
                "nonzero_contact_rows": nonzero_contact_count,
                "moment_rejected_rows": contact_moment_reject_count,
                "moment_rejected_fraction": contact_moment_reject_count / max(len(contact_force_norms), 1),
                "mean_contact_force_norm": _mean(contact_force_norms),
                "mean_contact_moment_norm": _mean(contact_moment_norms),
            }

        self._context_metadata["dynamics_residual_summary"] = {
            "rows": len(rows),
            "equation_form": "residual = conservative_terms - non_conservative_terms",
            "ddq_source": self.cfg.ddq_source,
            "ranking_order": "mean residual norm, then p95 residual norm, then max residual norm",
            "ranked_candidate_residuals": candidate_summaries,
            "ranked_model_candidate_residuals": model_candidate_summaries,
            "ranked_diagnostic_control_residuals": diagnostic_control_summaries,
            "clean_sample_ranked_candidate_residuals": clean_candidate_summaries,
            "clean_sample_ranked_model_candidate_residuals": clean_model_candidate_summaries,
            "clean_sample_ranked_diagnostic_control_residuals": clean_diagnostic_control_summaries,
            "dirty_sample_ranked_candidate_residuals": dirty_candidate_summaries,
            "dirty_sample_ranked_model_candidate_residuals": dirty_model_candidate_summaries,
            "dirty_sample_ranked_diagnostic_control_residuals": dirty_diagnostic_control_summaries,
            "best_by_family": best_by_family,
            "best_model_candidate_by_family": best_model_by_family,
            "best_diagnostic_control_by_family": best_diagnostic_by_family,
            "diagnostics": diagnostics,
            "residual_threshold_percentages": {
                "comparison": ">",
                "thresholds": [10.0, 100.0, 300.0, 500.0],
                "current": current_residual_thresholds,
                "best_model_candidate_label": best_model_label,
                "best_model_candidate": best_model_residual_thresholds,
            },
            "kinematic_quality": quality_summary,
            "sysid_model_policy": self._sysid_model_policy(),
            "candidate_role_definitions": {
                "model_candidate": (
                    "physically plausible full-coordinate equation variant; keeps the force balance complete "
                    "and may change only a source/convention for an existing physical term"
                ),
                "diagnostic_control": (
                    "debug-only control that removes, flips, zeroes, or time-shifts physics; use it to explain "
                    "mismatches, not as an export recommendation"
                ),
            },
        }

        print("\n[ForrestDynamics] Debug residual experiments")
        print("  residual form: conservative_terms - non_conservative_terms")
        print(f"  ddq/inertia source: {self.cfg.ddq_source}")
        print("  current conservative: inertia + gravity_identification + coriolis + tendon")
        print("  current non-conservative: actuation_command + contact_identification + friction")
        print(
            "  gravity/Coriolis source: PhysX direct generalized-force APIs when available; "
            "compensation APIs as debug/fallback"
        )
        print("  contact_identification uses measured contact_validated/contact_force/contact_moment")
        print("  contact debug split: contact_normal + contact_friction = contact when detail sensors are available")
        print(f"  rows: {len(rows):,}")
        if quality_summary.get("count", 0):
            print(
                "  kinematic quality: "
                f"adjacent rows={quality_summary['count']} "
                f"clean={quality_summary['clean_count']} "
                f"dirty={quality_summary['dirty_count']} "
                f"q_step_p95={quality_summary['q_step_norm']['p95']:.3f} "
                f"raw_ddq_vs_fd_p95={quality_summary['ddq_backward_norm']['p95']:.3f} "
                f"quality_p95={quality_summary['quality_norm']['p95']:.3f}"
            )
            skipped_pairs = quality_summary.get("skipped_non_adjacent_pairs", 0)
            if skipped_pairs:
                print(f"    skipped non-adjacent quality pairs: {skipped_pairs}")

        print("  ranked full-coordinate physics candidates by mean residual, lower is better:")
        self._print_debug_candidate_ranking(model_candidate_summaries, limit=12, include_groups=True)

        print("  ranked diagnostic controls by mean residual (not model candidates; lower is explanatory only):")
        self._print_debug_candidate_ranking(diagnostic_control_summaries, limit=12, include_groups=True)

        print("  best full-coordinate physics candidate by family:")
        for family, summary in best_model_by_family.items():
            print(f"    {family:<18} -> {summary['label']} (mean={summary['mean']:.3f}, p95={summary['p95']:.3f})")

        print("  best diagnostic control by family:")
        for family, summary in best_diagnostic_by_family.items():
            print(f"    {family:<18} -> {summary['label']} (mean={summary['mean']:.3f}, p95={summary['p95']:.3f})")

        if current_residual_thresholds:
            print("  current residual distribution:")
            print(f"    {self._format_threshold_percentages(current_residual_thresholds)}")
            if best_model_label and best_model_label != "current":
                print(f"  best physics candidate residual distribution ({best_model_label}):")
                print(f"    {self._format_threshold_percentages(best_model_residual_thresholds)}")

        self._print_debug_candidate_subset(
            "cleanest kinematic samples / physics candidates",
            clean_model_candidate_summaries,
            limit=8,
        )
        self._print_debug_candidate_subset(
            "cleanest kinematic samples / diagnostic controls",
            clean_diagnostic_control_summaries,
            limit=8,
        )
        self._print_debug_candidate_subset(
            "dirtiest kinematic samples / physics candidates",
            dirty_model_candidate_summaries,
            limit=8,
        )
        self._print_debug_candidate_subset(
            "dirtiest kinematic samples / diagnostic controls",
            dirty_diagnostic_control_summaries,
            limit=8,
        )

        print("  diagnostic deltas and term magnitudes:")
        for name, summary in diagnostics.items():
            if name == "contact_moment_validation":
                print(
                    "    contact_moment_validation: "
                    f"nonzero_rows={summary['nonzero_contact_rows']} "
                    f"moment_rejected={summary['moment_rejected_rows']} "
                    f"fraction={summary['moment_rejected_fraction']:.3f} "
                    f"mean_force={summary['mean_contact_force_norm']:.3f} "
                    f"mean_moment={summary['mean_contact_moment_norm']:.3f}"
                )
                continue
            print(f"    {name:<45} mean={summary['mean']:10.3f} p95={summary['p95']:10.3f} max={summary['max']:10.3f}")

        best_summary = (
            model_candidate_summaries[0]
            if model_candidate_summaries
            else (candidate_summaries[0] if candidate_summaries else None)
        )
        best_label = best_summary["label"] if best_summary is not None else "none"
        worst = candidate_worst.get(best_label)
        if worst is not None:
            _, row, residual = worst
            print(
                f"  worst row for best physics candidate '{best_label}': "
                f"sample_id={int(row[0])} step={int(row[1])} env={int(row[2])} side={row[3]}"
            )
            print("  worst residual by coordinate:")
            coordinate_names = self._coordinate_names_for_stream(str(row[3]))
            for dof, value in enumerate(residual):
                coordinate_label = coordinate_names[dof] if dof < len(coordinate_names) else f"q{dof}"
                print(f"    q{dof:<2d} {coordinate_label:<36} {value:+10.3f}")
        return

    def _debug_residual_candidate_specs(self) -> list[dict[str, Any]]:
        raw_conservative = (("inertia", 1.0), ("gravity", 1.0), ("coriolis", 1.0), ("tendon", 1.0))
        common_conservative = (
            ("inertia", 1.0),
            ("gravity_identification", 1.0),
            ("coriolis", 1.0),
            ("tendon", 1.0),
        )
        common_external = (("actuation_command", 1.0), ("contact_identification", 1.0), ("friction", 1.0))
        no_contact_external = (("actuation_command", 1.0), ("friction", 1.0))
        measured_contact_external = (("actuation_command", 1.0), ("contact_validated", 1.0), ("friction", 1.0))
        selected_ddq_source = self.cfg.ddq_source
        specs: list[dict[str, Any]] = []

        def add(
            label: str,
            family: str,
            conservative_terms,
            non_conservative_terms,
            *,
            role: str = "diagnostic_control",
            note: str = "",
        ) -> None:
            specs.append(
                {
                    "label": label,
                    "family": family,
                    "conservative_terms": tuple(conservative_terms),
                    "non_conservative_terms": tuple(non_conservative_terms),
                    "role": role,
                    "note": note,
                }
            )

        add(
            "current",
            "baseline",
            common_conservative,
            common_external,
            role="model_candidate",
            note="exported full-coordinate sysid equation with corrected base gravity and measured contact",
        )
        add("no_contact", "contact", common_conservative, no_contact_external)
        add(
            "inertia_recording_interval",
            "inertia",
            (
                ("inertia_recording_interval", 1.0),
                ("gravity_identification", 1.0),
                ("coriolis", 1.0),
                ("tendon", 1.0),
            ),
            common_external,
            role="model_candidate" if selected_ddq_source == "recording_interval" else "diagnostic_control",
            note=(
                "finite-difference acceleration diagnostic; this matches sim_data ddq only when "
                "ddq_source='recording_interval'"
            ),
        )
        add(
            "inertia_raw_physx",
            "inertia",
            (("inertia_raw", 1.0), ("gravity_identification", 1.0), ("coriolis", 1.0), ("tendon", 1.0)),
            common_external,
            role="model_candidate" if selected_ddq_source == "physx_raw" else "diagnostic_control",
            note="raw PhysX acceleration convention; this matches sim_data ddq when ddq_source='physx_raw'",
        )
        add(
            "inertia_physx_base_recording_joints",
            "inertia_source",
            (
                ("inertia_physx_base_recording_joints", 1.0),
                ("gravity_identification", 1.0),
                ("coriolis", 1.0),
                ("tendon", 1.0),
            ),
            common_external,
            role="model_candidate",
            note=(
                "source split: raw PhysX base acceleration with recording-interval joint acceleration; "
                "physically valid only if sim_data ddq exports the same split source"
            ),
        )
        add(
            "inertia_recording_base_physx_joints",
            "inertia_source",
            (
                ("inertia_recording_base_physx_joints", 1.0),
                ("gravity_identification", 1.0),
                ("coriolis", 1.0),
                ("tendon", 1.0),
            ),
            common_external,
            role="model_candidate",
            note=(
                "source split: recording-interval base acceleration with raw PhysX joint acceleration; "
                "physically valid only if sim_data ddq exports the same split source"
            ),
        )
        add(
            "inertia_physx_base_body_frame",
            "base_acceleration_convention",
            (
                ("inertia_physx_base_body_frame", 1.0),
                ("gravity_identification", 1.0),
                ("coriolis", 1.0),
                ("tendon", 1.0),
            ),
            common_external,
            role="model_candidate",
            note=(
                "frame convention check: rotates raw PhysX base linear/angular acceleration from world to root frame "
                "before M(q)ddq; valid only if exported base dq/ddq and forces use the same convention"
            ),
        )
        add(
            "inertia_physx_base_swapped",
            "base_acceleration_convention",
            (
                ("inertia_physx_base_swapped", 1.0),
                ("gravity_identification", 1.0),
                ("coriolis", 1.0),
                ("tendon", 1.0),
            ),
            common_external,
            role="model_candidate",
            note=(
                "ordering convention check: treats the raw PhysX base acceleration vector as [angular, linear] "
                "instead of [linear, angular]; valid only if the API/metadata convention is confirmed wrong"
            ),
        )
        add(
            "old_raw_inertia_with_measured_contact",
            "legacy_baseline",
            (("inertia_raw", 1.0), ("gravity", 1.0), ("coriolis", 1.0), ("tendon", 1.0)),
            measured_contact_external,
        )
        add(
            "inertia_joint_only",
            "inertia",
            (("inertia_joint_only", 1.0), ("gravity_identification", 1.0), ("coriolis", 1.0), ("tendon", 1.0)),
            common_external,
        )
        add(
            "gravity_coriolis_force_api",
            "gravity_coriolis_api",
            (("inertia", 1.0), ("gravity_force_api", 1.0), ("coriolis_force_api", 1.0), ("tendon", 1.0)),
            common_external,
            role="model_candidate",
            note=(
                "uses PhysX direct generalized-force aliases for gravity and Coriolis when available; "
                "compensation-derived aliases only when direct APIs are absent"
            ),
        )
        add(
            "gravity_coriolis_compensation_actual",
            "gravity_coriolis_api",
            (
                ("inertia", 1.0),
                ("gravity_compensation_actual", 1.0),
                ("coriolis_compensation_actual", 1.0),
                ("tendon", 1.0),
            ),
            common_external,
            role="model_candidate",
            note="uses compensation APIs converted to actual generalized forces",
        )
        add(
            "coriolis_compensation_actual_current_gravity",
            "coriolis_source",
            (
                ("inertia", 1.0),
                ("gravity_identification", 1.0),
                ("coriolis_compensation_actual", 1.0),
                ("tendon", 1.0),
            ),
            common_external,
            role="model_candidate",
            note=(
                "isolates the Coriolis source choice while keeping the current base-gravity identification "
                "convention unchanged"
            ),
        )
        add(
            "external_comp_gravity_contact",
            "base_force_convention",
            raw_conservative,
            (
                ("actuation_command", 1.0),
                ("gravity_compensation_actual", 1.0),
                ("contact_validated", 1.0),
                ("friction", 1.0),
            ),
            note=(
                "diagnostic only: treats measured contact plus full compensation-derived gravity as external "
                "world-frame wrenches. This can move joint gravity terms across sides, so the final sysid export "
                "uses gravity_identification instead."
            ),
        )
        add(
            "external_base_gravity_full_contact",
            "base_force_convention",
            raw_conservative,
            (
                ("actuation_command", 1.0),
                ("gravity_compensation_actual", 1.0, "base"),
                ("contact_validated", 1.0),
                ("friction", 1.0),
            ),
            role="model_candidate",
            note=(
                "applies compensation-derived gravity only to base coordinates while keeping full measured "
                "contact projection"
            ),
        )
        add(
            "external_base_gravity_base_contact",
            "base_force_convention",
            raw_conservative,
            (
                ("actuation_command", 1.0),
                ("gravity_compensation_actual", 1.0, "base"),
                ("contact_validated", 1.0, "base"),
                ("friction", 1.0),
            ),
            note=(
                "base-only diagnostic: tests whether the residual reduction comes from the floating-base "
                "wrench convention"
            ),
        )
        add(
            "external_base_gravity_no_contact",
            "base_force_convention",
            raw_conservative,
            (
                ("actuation_command", 1.0),
                ("gravity_compensation_actual", 1.0, "base"),
                ("friction", 1.0),
            ),
            note="base-only diagnostic: isolates compensation-derived gravity without measured contact",
        )
        add(
            "external_base_contact_no_gravity",
            "base_force_convention",
            raw_conservative,
            (
                ("actuation_command", 1.0),
                ("contact_validated", 1.0, "base"),
                ("friction", 1.0),
            ),
            note="base-only diagnostic: isolates measured contact without compensation-derived gravity",
        )
        add(
            "external_comp_gravity_contact_no_coriolis",
            "base_force_convention",
            (("inertia", 1.0), ("gravity", 1.0), ("tendon", 1.0)),
            (
                ("actuation_command", 1.0),
                ("gravity_compensation_actual", 1.0),
                ("contact_validated", 1.0),
                ("friction", 1.0),
            ),
        )
        add(
            "no_coriolis",
            "bias_sign",
            (("inertia", 1.0), ("gravity_identification", 1.0), ("tendon", 1.0)),
            common_external,
        )
        add(
            "no_coriolis_no_tendon",
            "term_removal_diagnostic",
            (("inertia", 1.0), ("gravity_identification", 1.0)),
            common_external,
        )
        add(
            "inertia_only",
            "term_removal_diagnostic",
            (("inertia", 1.0),),
            common_external,
        )
        add(
            "flip_coriolis_sign",
            "bias_sign",
            (("inertia", 1.0), ("gravity_identification", 1.0), ("coriolis", -1.0), ("tendon", 1.0)),
            common_external,
        )
        add("no_gravity", "bias_sign", (("inertia", 1.0), ("coriolis", 1.0), ("tendon", 1.0)), common_external)
        add(
            "flip_gravity_sign",
            "bias_sign",
            (("inertia", 1.0), ("gravity_identification", -1.0), ("coriolis", 1.0), ("tendon", 1.0)),
            common_external,
        )
        add(
            "tendon_model",
            "tendon",
            (
                ("inertia", 1.0),
                ("gravity_identification", 1.0),
                ("coriolis", 1.0),
                ("tendon_model", 1.0),
            ),
            common_external,
            role="model_candidate",
            note="uses analytic joint-space tendon model instead of projected link-wrench tendon term",
        )
        add(
            "no_tendon",
            "tendon",
            (("inertia", 1.0), ("gravity_identification", 1.0), ("coriolis", 1.0)),
            common_external,
        )
        add(
            "half_tendon_lhs",
            "tendon",
            (("inertia", 1.0), ("gravity_identification", 1.0), ("coriolis", 1.0), ("tendon", 0.5)),
            common_external,
        )
        add(
            "flip_tendon_sign",
            "tendon",
            (("inertia", 1.0), ("gravity_identification", 1.0), ("coriolis", 1.0), ("tendon", -1.0)),
            common_external,
        )
        add(
            "contact_force_only",
            "contact",
            common_conservative,
            (("actuation_command", 1.0), ("contact_force", 1.0), ("friction", 1.0)),
        )
        add(
            "contact_normal_only",
            "contact",
            common_conservative,
            (("actuation_command", 1.0), ("contact_normal", 1.0), ("friction", 1.0)),
            note="diagnostic: drops measured tangential/friction contact to quantify its physical contribution",
        )
        add(
            "contact_friction_opposite_sign",
            "contact",
            common_conservative,
            (("actuation_command", 1.0), ("contact_normal", 1.0), ("contact_friction", -1.0), ("friction", 1.0)),
            note="diagnostic check for tangential contact sign convention",
        )
        add(
            "contact_force_opposite_moment",
            "contact",
            common_conservative,
            (("actuation_command", 1.0), ("contact_force", 1.0), ("contact_moment", -1.0), ("friction", 1.0)),
            note="diagnostic check for contact lever-arm moment sign convention",
        )
        add(
            "measured_contact_validated",
            "contact",
            common_conservative,
            measured_contact_external,
            role="model_candidate",
            note="uses measured contact force plus accepted contact-point moment",
        )
        add(
            "contact_full_force_plus_moment",
            "contact",
            common_conservative,
            (("actuation_command", 1.0), ("contact", 1.0), ("friction", 1.0)),
            role="model_candidate",
            note="uses measured contact force plus moment without moment rejection",
        )
        add(
            "contact_groups_digit_connector_base_self",
            "contact",
            common_conservative,
            (
                ("actuation_command", 1.0),
                ("contact_digit", 1.0),
                ("contact_connector", 1.0),
                ("contact_base", 1.0),
                ("contact_self_collision", 1.0),
                ("friction", 1.0),
            ),
            role="model_candidate",
            note="uses the sum of measured contact groups to detect omitted contact bodies",
        )
        add(
            "flip_measured_contact_sign",
            "contact",
            common_conservative,
            (("actuation_command", 1.0), ("contact_validated", -1.0), ("friction", 1.0)),
        )
        add(
            "motor_actuation",
            "actuation",
            common_conservative,
            (("motor_actuation", 1.0), ("contact_identification", 1.0), ("friction", 1.0)),
            role="model_candidate",
            note="uses all recorded motor torques as the actuation source",
        )
        add(
            "physx_actuation",
            "actuation",
            common_conservative,
            (("physx_actuation", 1.0), ("contact_identification", 1.0), ("friction", 1.0)),
        )
        add(
            "command_plus_pantograph",
            "actuation",
            common_conservative,
            (
                ("actuation_command", 1.0),
                ("pantograph_actuation", 1.0),
                ("contact_identification", 1.0),
                ("friction", 1.0),
            ),
            role="model_candidate",
            note="adds explicit pantograph actuation diagnostic term to the command source",
        )
        add(
            "implicit_drive_estimate",
            "actuation",
            common_conservative,
            (("implicit_drive_estimate", 1.0), ("contact_identification", 1.0), ("friction", 1.0)),
            note=(
                "diagnostic: uses the PD force that implicit PhysX drives are commanded to solve, clipped by "
                "joint effort limits; this can include passive/constraint-like drive coordinates"
            ),
        )
        add(
            "command_plus_knee_flexor",
            "actuation",
            common_conservative,
            (
                ("actuation_command", 1.0),
                ("knee_flexor_actuation", 1.0),
                ("contact_identification", 1.0),
                ("friction", 1.0),
            ),
            role="model_candidate",
            note="adds explicit knee-flexor actuation diagnostic term to the command source",
        )
        add(
            "command_plus_pantograph_plus_knee",
            "actuation",
            common_conservative,
            (
                ("actuation_command", 1.0),
                ("pantograph_actuation", 1.0),
                ("knee_flexor_actuation", 1.0),
                ("contact_identification", 1.0),
                ("friction", 1.0),
            ),
            role="model_candidate",
            note="adds explicit pantograph and knee-flexor actuation terms to the command source",
        )
        add(
            "command_plus_solver_internal",
            "solver",
            common_conservative,
            (*common_external, ("solver_constraint_internal", 1.0)),
            note=(
                "diagnostic only: treats passive/limit solver reactions as external generalized constraint forces; "
                "do not use as the sysid model unless those reactions are mapped to a physical constraint law"
            ),
        )
        add(
            "command_minus_solver_internal",
            "solver",
            common_conservative,
            (*common_external, ("solver_constraint_internal", -1.0)),
        )
        add(
            "physx_plus_solver_internal",
            "solver",
            common_conservative,
            (
                ("physx_actuation", 1.0),
                ("contact_identification", 1.0),
                ("friction", 1.0),
                ("solver_constraint_internal", 1.0),
            ),
        )
        add(
            "no_friction",
            "friction",
            common_conservative,
            (("actuation_command", 1.0), ("contact_identification", 1.0)),
        )
        return specs

    def _coordinate_report_groups(self) -> dict[str, dict[str, list[int]]]:
        groups: dict[str, dict[str, list[int]]] = {}
        for side in self._selected_sides():
            names = self._coordinate_names_for_stream(side)
            side_groups: dict[str, list[int]] = {}
            if self._base_coordinate_count_by_side.get(side, 0) >= BASE_COORDINATE_COUNT:
                side_groups["base"] = list(range(BASE_COORDINATE_COUNT))
                side_groups["base_linear"] = [0, 1, 2]
                side_groups["base_angular"] = [3, 4, 5]
            left = [index for index, name in enumerate(names) if name.startswith("l")]
            right = [index for index, name in enumerate(names) if name.startswith("r")]
            joints = [index for index, name in enumerate(names) if name.startswith(("l", "r"))]
            if left:
                side_groups["left_leg"] = left
            if right:
                side_groups["right_leg"] = right
            if joints:
                side_groups["all_joints"] = joints
            side_groups["all"] = list(range(self.num_dofs))
            groups[side] = side_groups
        return groups

    def _all_report_group_names(self, coordinate_groups: dict[str, dict[str, list[int]]]) -> tuple[str, ...]:
        names = sorted({group_name for groups in coordinate_groups.values() for group_name in groups})
        return tuple(names)

    def _format_group_suffix(self, group_mean_norms: dict[str, float]) -> str:
        selected = []
        for group_name in ("base_linear", "base_angular", "left_leg", "right_leg"):
            if group_name in group_mean_norms:
                selected.append(f"{group_name}={group_mean_norms[group_name]:.2f}")
        if not selected:
            return ""
        return " | " + ", ".join(selected)

    def _debug_candidate_summaries(
        self,
        candidate_norms: dict[str, list[float]],
        *,
        candidate_meta: dict[str, dict[str, Any]],
        candidate_group_norms: dict[str, dict[str, list[float]]] | None = None,
    ) -> list[dict[str, Any]]:
        summaries = []
        for label, norms in candidate_norms.items():
            if not norms:
                continue
            summary = {
                "label": label,
                "family": candidate_meta[label]["family"],
                "role": candidate_meta[label].get("role", "diagnostic_control"),
                "note": candidate_meta[label].get("note", ""),
                **_norm_summary(norms),
            }
            if candidate_group_norms is not None:
                summary["group_mean_norms"] = {
                    group_name: _mean(group_norms)
                    for group_name, group_norms in candidate_group_norms[label].items()
                    if group_norms
                }
            summaries.append(summary)
        summaries.sort(key=lambda item: (item["mean"], item["p95"], item["max"]))
        return summaries

    def _summaries_for_role(self, summaries: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        return [summary for summary in summaries if summary.get("role") == role]

    def _best_by_family(self, summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        best_by_family: dict[str, dict[str, Any]] = {}
        for summary in summaries:
            best_by_family.setdefault(summary["family"], summary)
        return best_by_family

    def _print_debug_candidate_ranking(
        self,
        summaries: list[dict[str, Any]],
        *,
        limit: int,
        include_groups: bool = False,
    ) -> None:
        if not summaries:
            print("    none")
            return
        for rank, summary in enumerate(summaries[:limit], start=1):
            group_suffix = self._format_group_suffix(summary.get("group_mean_norms", {})) if include_groups else ""
            print(
                f"    {rank:2d}. {summary['label']:<42} "
                f"family={summary['family']:<18} mean={summary['mean']:10.3f} "
                f"p95={summary['p95']:10.3f} max={summary['max']:10.3f}{group_suffix}"
            )

    def _print_debug_candidate_subset(self, title: str, summaries: list[dict[str, Any]], *, limit: int) -> None:
        if not summaries:
            return
        print(f"  ranked candidate equations on {title}:")
        self._print_debug_candidate_ranking(summaries, limit=limit, include_groups=False)

    def _format_threshold_percentages(self, threshold_rows: list[dict[str, float | int | str]]) -> str:
        return ", ".join(f">{row['threshold']:.0f}: {row['percent_above']:.2f}%" for row in threshold_rows)

    def _debug_sample_kinematic_quality(
        self,
        debug_rows: list[tuple[Any, ...]],
        col_index: dict[str, int],
    ) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
        if self._db is None or self._sim_dt is None or self._sim_dt <= 0.0 or self.num_dofs <= 0:
            return {}, {}

        sim_columns = _sim_data_columns(self.num_dofs)
        quoted_cols = ", ".join(_quote_identifier(name) for name in sim_columns)
        sim_rows = self._db.execute(
            f"SELECT rowid - 1 AS sample_id, {quoted_cols} "
            f"FROM {_quote_identifier(self.cfg.sim_table_name)} ORDER BY rowid"
        ).fetchall()
        sim_by_sample_id = {int(row[0]): tuple(float(value) for value in row[1:]) for row in sim_rows}
        if not sim_by_sample_id:
            return {}, {}

        samples_by_stream: dict[tuple[int, str], list[tuple[int, int]]] = {}
        for row in debug_rows:
            sample_id = int(row[col_index["sample_id"]])
            if sample_id not in sim_by_sample_id:
                continue
            key = (int(row[col_index["env_id"]]), str(row[col_index["side"]]))
            samples_by_stream.setdefault(key, []).append((int(row[col_index["step_index"]]), sample_id))

        quality: dict[int, dict[str, float]] = {}
        skipped_non_adjacent_pairs = 0
        for (_env_id, side), samples in samples_by_stream.items():
            samples.sort()
            angular_indices = set(self._angular_coordinate_indices_for_stream(side))
            base_count = self._base_coordinate_count_by_side.get(side, 0)
            base_euler_indices = set(range(3, min(base_count, BASE_COORDINATE_COUNT)))
            for local_index in range(1, len(samples)):
                prev_step, prev_sample_id = samples[local_index - 1]
                step, sample_id = samples[local_index]
                step_delta = step - prev_step
                if step_delta != 1:
                    skipped_non_adjacent_pairs += 1
                    continue
                dt = float(step_delta) * float(self._sim_dt)
                if dt <= 0.0:
                    continue
                prev_values = sim_by_sample_id[prev_sample_id]
                values = sim_by_sample_id[sample_id]
                q_step_error = []
                ddq_backward_error = []
                for dof_index in range(self.num_dofs):
                    if dof_index not in base_euler_indices:
                        q_delta = values[dof_index] - prev_values[dof_index]
                        if dof_index in angular_indices:
                            q_delta = _wrap_to_pi(q_delta)
                        q_step_error.append(
                            q_delta
                            - 0.5 * (prev_values[self.num_dofs + dof_index] + values[self.num_dofs + dof_index]) * dt
                        )
                    dq_delta = values[self.num_dofs + dof_index] - prev_values[self.num_dofs + dof_index]
                    ddq_backward_error.append(values[2 * self.num_dofs + dof_index] - dq_delta / dt)
                q_step_norm = _vector_norm(tuple(q_step_error))
                ddq_backward_norm = _vector_norm(tuple(ddq_backward_error))
                quality[sample_id] = {
                    "q_step_norm": q_step_norm,
                    "ddq_backward_norm": ddq_backward_norm,
                    "quality_norm": q_step_norm,
                }
        return quality, {
            "policy": (
                "adjacent one-step check against exported sim_data: q_k - q_{k-1} is compared with trapezoidal "
                "dq integration. ddq_k is also compared with backward velocity difference as a diagnostic, but "
                "that ddq diagnostic is not used for clean/dirty splitting when raw PhysX ddq is exported."
            ),
            "q_step_exclusion": (
                "floating-base Euler orientation coordinates are excluded from q-step integration because their "
                "dq slots store world angular velocity, not Euler-angle rates"
            ),
            "skipped_non_adjacent_pairs": skipped_non_adjacent_pairs,
        }

    def _debug_quality_subsets(
        self,
        sample_quality: dict[int, dict[str, float]],
        *,
        quality_pair_summary: dict[str, Any],
    ) -> tuple[set[int], set[int], dict[str, Any]]:
        if not sample_quality:
            return set(), set(), dict(quality_pair_summary)
        quality_values = [quality["quality_norm"] for quality in sample_quality.values()]
        clean_cutoff = _percentile(quality_values, 25.0)
        dirty_cutoff = _percentile(quality_values, 95.0)
        clean_sample_ids = {
            sample_id for sample_id, quality in sample_quality.items() if quality["quality_norm"] <= clean_cutoff
        }
        dirty_sample_ids = {
            sample_id
            for sample_id, quality in sample_quality.items()
            if quality["quality_norm"] >= dirty_cutoff and sample_id not in clean_sample_ids
        }
        return (
            clean_sample_ids,
            dirty_sample_ids,
            {
                **quality_pair_summary,
                "count": len(sample_quality),
                "clean_count": len(clean_sample_ids),
                "dirty_count": len(dirty_sample_ids),
                "clean_cutoff": clean_cutoff,
                "dirty_cutoff": dirty_cutoff,
                "q_step_norm": _norm_summary([quality["q_step_norm"] for quality in sample_quality.values()]),
                "ddq_backward_norm": _norm_summary(
                    [quality["ddq_backward_norm"] for quality in sample_quality.values()]
                ),
                "quality_norm": _norm_summary(quality_values),
            },
        )

    def _angular_coordinate_indices_for_stream(self, side: str) -> tuple[int, ...]:
        angular_indices = []
        base_count = self._base_coordinate_count_by_side.get(side, 0)
        if base_count:
            angular_indices.extend(range(3, min(base_count, BASE_COORDINATE_COUNT)))
        angular_indices.extend(range(base_count, self.num_dofs))
        return tuple(angular_indices)

    def _sim_units(self) -> dict[str, Any]:
        if not any(self._base_coordinate_count_by_side.get(side, 0) for side in self._selected_sides()):
            return {"q": "rad", "dq": "rad/s", "ddq": "rad/s^2", "tau": "N*m"}
        return {
            "q": "per-coordinate; see coordinate_mappings[].q_unit",
            "dq": "per-coordinate; see coordinate_mappings[].dq_unit",
            "ddq": "per-coordinate; see coordinate_mappings[].ddq_unit",
            "tau": "per-coordinate generalized force; see coordinate_mappings[].tau_unit",
        }

    def _coordinate_names_for_stream(self, side: str) -> tuple[str, ...]:
        base_names = BASE_COORDINATE_NAMES if self._base_coordinate_count_by_side.get(side, 0) else ()
        return (*base_names, *self._joint_names_by_side.get(side, ()))

    def _coordinate_metadata(self) -> list[dict[str, Any]]:
        rows = []
        for side in self._selected_sides():
            q_index = 0
            if self._base_coordinate_count_by_side.get(side, 0):
                for local_index, name in enumerate(BASE_COORDINATE_NAMES):
                    rows.append(
                        {
                            "side": side,
                            "q_index": q_index,
                            "coordinate_name": name,
                            "coordinate_type": "floating_base",
                            "isaac_joint_index": None,
                            "q_unit": BASE_COORDINATE_UNITS[local_index],
                            "dq_unit": BASE_VELOCITY_UNITS[local_index],
                            "ddq_unit": BASE_ACCELERATION_UNITS[local_index],
                            "ddq_source": self.cfg.ddq_source,
                            "tau_unit": BASE_FORCE_UNITS[local_index],
                            "frame": "world",
                            "sign_convention": "isaac_root_state",
                            "offset_convention": "raw_isaac_root_pose",
                        }
                    )
                    q_index += 1
            for joint_name, joint_index in zip(self._joint_names_by_side[side], self._joint_indices_by_side[side]):
                rows.append(
                    {
                        "side": side,
                        "q_index": q_index,
                        "coordinate_name": joint_name,
                        "coordinate_type": "joint",
                        "joint_name": joint_name,
                        "isaac_joint_index": int(joint_index),
                        "q_unit": "rad",
                        "dq_unit": "rad/s",
                        "ddq_unit": "rad/s^2",
                        "ddq_source": self.cfg.ddq_source,
                        "tau_unit": "N*m",
                        "sign_convention": "isaac_joint_position",
                        "offset_convention": "raw_isaac_joint_position",
                    }
                )
                q_index += 1
        return rows

    def _joint_metadata(self) -> list[dict[str, Any]]:
        rows = []
        for side, names in self._joint_names_by_side.items():
            q_offset = self._base_coordinate_count_by_side.get(side, 0)
            for q_index, (joint_name, joint_index) in enumerate(zip(names, self._joint_indices_by_side[side])):
                rows.append(
                    {
                        "side": side,
                        "q_index": q_offset + q_index,
                        "joint_name": joint_name,
                        "isaac_joint_index": int(joint_index),
                        "units": "rad",
                        "sign_convention": "isaac_joint_position",
                        "offset_convention": "raw_isaac_joint_position",
                    }
                )
        return rows

    def _joint_dynamics_properties(self) -> list[dict[str, Any]]:
        return list(self._joint_dynamics_properties_rows)

    def _body_metadata(self) -> list[dict[str, Any]]:
        rows = []
        for side, names in self._body_names_by_side.items():
            for body_name, body_index in zip(names, self._body_indices_by_side[side]):
                rows.append(
                    {
                        "side": side,
                        "body_name": body_name,
                        "isaac_body_index": int(body_index),
                        "frame": "world",
                    }
                )
        return rows

    def _body_dynamics_properties(self) -> list[dict[str, Any]]:
        return list(self._body_dynamics_properties_rows)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _norm_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "max": max(float(value) for value in values),
    }


def _threshold_percentages(
    values: list[float],
    thresholds: tuple[float, ...] = (10.0, 100.0, 300.0, 500.0),
) -> list[dict[str, float | int | str]]:
    if not values:
        return []
    count = len(values)
    rows = []
    for threshold in thresholds:
        above = sum(1 for value in values if float(value) > threshold)
        rows.append(
            {
                "comparison": ">",
                "threshold": threshold,
                "rows_above": above,
                "total_rows": count,
                "fraction_above": above / count,
                "percent_above": 100.0 * above / count,
            }
        )
    return rows


def _vector_norm(values: tuple[float, ...], indices: list[int] | None = None) -> float:
    selected = values if indices is None else tuple(values[index] for index in indices)
    if not selected:
        return 0.0
    return float(sum(float(value) * float(value) for value in selected) ** 0.5)


def _vector_add(*vectors: tuple[float, ...]) -> tuple[float, ...]:
    if not vectors:
        return ()
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("Cannot add vectors with different widths.")
    return tuple(sum(float(vector[index]) for vector in vectors) for index in range(width))


def _vector_sub(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("Cannot subtract vectors with different widths.")
    return tuple(float(left[index]) - float(right[index]) for index in range(len(left)))


def _weighted_vector_sum(
    terms: dict[str, tuple[float, ...]],
    weighted_terms: tuple[tuple[Any, ...], ...],
    width: int,
    *,
    coordinate_masks: dict[str, list[int]] | None = None,
) -> tuple[float, ...]:
    result = [0.0] * width
    for weighted_term in weighted_terms:
        term_name = _weighted_term_name(weighted_term)
        scale = float(weighted_term[1])
        mask_name = str(weighted_term[2]) if len(weighted_term) >= 3 else None
        values = terms[term_name]
        if len(values) != width:
            raise ValueError(f"Term {term_name!r} has width {len(values)}, expected {width}.")
        if mask_name is None:
            indices = range(width)
        else:
            if coordinate_masks is None or mask_name not in coordinate_masks:
                raise KeyError(f"Unknown coordinate mask {mask_name!r} for term {term_name!r}.")
            indices = coordinate_masks[mask_name]
        for index in indices:
            value = values[index]
            result[index] += scale * float(value)
    return tuple(result)


def _weighted_term_name(weighted_term: tuple[Any, ...]) -> str:
    if len(weighted_term) < 2:
        raise ValueError(f"Weighted term must contain at least name and weight: {weighted_term!r}")
    return str(weighted_term[0])


def _wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _sim_data_columns(num_dofs: int) -> list[str]:
    return (
        [f"q{i}" for i in range(num_dofs)]
        + [f"dq{i}" for i in range(num_dofs)]
        + [f"ddq{i}" for i in range(num_dofs)]
        + [f"tau{i}" for i in range(num_dofs)]
    )


def _training_dynamics_data_columns(num_dofs: int) -> list[str]:
    return ["sample_id", "step_index", "time", "env_id", "side"] + [
        f"tau_{name}{i}" for name in TRAINING_DYNAMICS_TERM_NAMES for i in range(num_dofs)
    ]


def _debug_dynamics_data_columns(
    num_dofs: int,
    *,
    matrix_term_names: tuple[str, ...] = DEBUG_DYNAMICS_MATRIX_TERM_NAMES,
) -> list[str]:
    return (
        ["sample_id", "step_index", "time", "env_id", "side"]
        + [f"tau_{name}{i}" for name in DEBUG_DYNAMICS_TERM_NAMES for i in range(num_dofs)]
        + list(DEBUG_DYNAMICS_SCALAR_NAMES)
        + [
            f"{term_name}{row}_{col}"
            for term_name in matrix_term_names
            for row in range(num_dofs)
            for col in range(num_dofs)
        ]
    )


def _debug_matrix_term_names_for_num_dofs(num_dofs: int) -> tuple[str, ...]:
    column_count = 5 + len(DEBUG_DYNAMICS_TERM_NAMES) * num_dofs + len(DEBUG_DYNAMICS_SCALAR_NAMES)
    selected = []
    for term_name in DEBUG_DYNAMICS_MATRIX_TERM_NAMES:
        term_columns = num_dofs * num_dofs
        if column_count + term_columns > SQLITE_DEBUG_COLUMN_LIMIT:
            continue
        selected.append(term_name)
        column_count += term_columns
    return tuple(selected)


def _dynamics_column_sql(name: str) -> str:
    if name == "sample_id":
        return f"{_quote_identifier(name)} INTEGER PRIMARY KEY"
    if name in ("step_index", "env_id"):
        return f"{_quote_identifier(name)} INTEGER NOT NULL"
    if name == "side":
        return f"{_quote_identifier(name)} TEXT NOT NULL"
    return f"{_quote_identifier(name)} REAL NOT NULL"


def _spatial_columns() -> list[str]:
    names = ["sample_id", "step_index", "time", "env_id", "side", "body_index", "body_name"]
    names += _named_columns("root_pos", ("x", "y", "z"))
    names += _named_columns("root_quat", ("w", "x", "y", "z"))
    names += _named_columns("root_lin_vel", ("x", "y", "z"))
    names += _named_columns("root_ang_vel", ("x", "y", "z"))
    names += _named_columns("body_link_pos", ("x", "y", "z"))
    names += _named_columns("body_link_quat", ("w", "x", "y", "z"))
    names += _named_columns("body_link_lin_vel", ("x", "y", "z"))
    names += _named_columns("body_link_ang_vel", ("x", "y", "z"))
    names += _named_columns("body_com_pos", ("x", "y", "z"))
    names += _named_columns("body_com_quat", ("w", "x", "y", "z"))
    names += _named_columns("body_com_lin_vel", ("x", "y", "z"))
    names += _named_columns("body_com_ang_vel", ("x", "y", "z"))
    names += _named_columns("body_com_lin_acc", ("x", "y", "z"))
    names += _named_columns("body_com_ang_acc", ("x", "y", "z"))
    return names


def _named_columns(prefix: str, suffixes: tuple[str, ...]) -> list[str]:
    return [f"{prefix}_{suffix}" for suffix in suffixes]


def _spatial_column_sql(name: str) -> str:
    if name in ("sample_id", "step_index", "env_id", "body_index"):
        return f"{_quote_identifier(name)} INTEGER NOT NULL"
    if name in ("side", "body_name"):
        return f"{_quote_identifier(name)} TEXT NOT NULL"
    return f"{_quote_identifier(name)} REAL NOT NULL"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _validate_unique(values: tuple[str, ...], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique: {values}")


def _to_int_list(values) -> list[int]:
    return [int(value) for value in values]


def _tensor_scalar(tensor, joint_index: int) -> float:
    value = tensor[0, joint_index] if tensor.ndim == 2 else tensor[joint_index]
    return float(value.detach().cpu().item())


def _jsonable_scalar(value: float) -> float | str:
    value = float(value)
    if math.isfinite(value):
        return value
    if value > 0:
        return "inf"
    if value < 0:
        return "-inf"
    return "nan"


def _normalize_optional_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in ("", "none", "null"):
            return None
        value = stripped
    return float(value)


def motor_torque_tensor(robot):
    """Return applied motor torque with all non-motor joints explicitly zeroed."""

    tau = robot.data.applied_torque * 0.0
    motor_names = set(actuated_joint_names)
    motor_indices = [index for index, joint_name in enumerate(robot.joint_names) if joint_name in motor_names]
    if motor_indices:
        tau[:, motor_indices] = robot.data.applied_torque[:, motor_indices]
    return tau


def actuation_command_tensor(robot):
    """Return the actuation term used by the current sysid force balance."""

    tau = motor_torque_tensor(robot)
    knee_flexor_indices = [
        index for index, joint_name in enumerate(robot.joint_names) if joint_name in KNEE_FLEXOR_JOINT_NAMES
    ]
    if knee_flexor_indices:
        tau[:, knee_flexor_indices] = 0.0
    return tau


def _jsonable_config(cfg: DataRecordingConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["output_dir"] = str(data["output_dir"])
    return data
