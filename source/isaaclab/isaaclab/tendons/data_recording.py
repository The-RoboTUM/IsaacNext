# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identix-compatible data recording helpers for Forrest tendon simulations."""

from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.tendons.models.analytic.constants import (
    joint_names_left,
    joint_names_right,
    link_names_left,
    link_names_right,
)

TENDON_CHAIN_5_JOINTS: dict[str, tuple[str, ...]] = {
    "left": tuple(joint_names_left),
    "right": tuple(joint_names_right),
}
REAL_LEG_JOINTS: dict[str, tuple[str, ...]] = {
    "left": (
        "lp1_pantograph",
        "l0_acetabulofemoral_roll",
        "l1_acetabulofemoral_lateral",
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
        "rp1_pantograph",
        "r0_acetabulofemoral_roll",
        "r1_acetabulofemoral_lateral",
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

CONTACT_GROUP_NAMES = ("digit", "connector", "base")

TRAINING_DYNAMICS_TERM_NAMES = (
    "inertia",
    "coriolis",
    "gravity",
    "tendon",
    "actuation",
    "contact",
    "friction",
    "external",
)

DEBUG_DYNAMICS_TERM_NAMES = (
    "inertia",
    "inertia_recording_interval",
    "inertia_raw",
    "inertia_joint_only",
    "inertia_joint_all",
    "inertia_leg_self",
    "inertia_other_joints",
    "inertia_root_coupling",
    "inertia_root_coupling_raw",
    "inertia_root_coupling_alt",
    "inertia_root_coupled_alt",
    "inertia_full_raw",
    "coriolis",
    "gravity",
    "friction_dynamic",
    "friction_viscous",
    "friction",
    "solver_joint",
    "actuation",
    "actuation_command",
    "actuation_estimated",
    "actuation_estimated_hip",
    "actuation_estimated_hip_lateral_flexion",
    "actuation_estimated_passive",
    "physx_actuation",
    "solver_constraint_passive",
    "solver_constraint_limit",
    "solver_constraint_internal",
    "joint_drive_pos_target",
    "joint_drive_vel_target",
    "joint_drive_effort_target",
    "joint_drive_stiffness",
    "joint_drive_damping",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_limit_distance_lower",
    "joint_limit_distance_upper",
    "joint_limit_distance_min",
    "drive_stiffness",
    "drive_damping",
    "drive_effort_target",
    "drive_pd",
    "drive_pd_clipped",
    "armature_inertia",
    "contact",
    "contact_force",
    "contact_moment",
    "contact_validated",
    "contact_digit",
    "contact_digit_force",
    "contact_digit_moment",
    "contact_connector",
    "contact_connector_force",
    "contact_connector_moment",
    "contact_base",
    "contact_base_force",
    "contact_base_moment",
    "tendon",
    "tendon_model",
    "tendon_projection_delta",
    "unmodeled_quasistatic",
    "unmodeled_full_dynamics",
    "unmodeled_recording_interval",
    "unmodeled_estimated_actuation",
    "unmodeled_estimated_hip_actuation",
    "unmodeled_estimated_hip_force_contact",
    "unmodeled_estimated_hip_force_contact_solver",
    "unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal",
    "unmodeled_full_contact",
    "unmodeled_contact_force_only",
    "unmodeled_contact_validated",
    "unmodeled",
    "inverse_residual",
    "solver_residual",
)
DEBUG_DYNAMICS_MATRIX_TERM_NAMES = ("mass_matrix",)
DEBUG_DYNAMICS_SCALAR_NAMES = (
    "quality_primary_residual_norm",
    "quality_sysid_residual_norm",
    "quality_sysid_residual_pct",
    "quality_dynamics_scale",
    "quality_usable_100",
    "quality_usable_150",
    "quality_usable_200",
    "quality_contact_norm",
    "quality_solver_internal_norm",
    "quality_inertia_norm",
    "quality_command_norm",
    "quality_max_limit_penetration",
    "quality_has_limit_penetration",
    "quality_has_inertia_spike",
    "quality_has_solver_spike",
    "quality_has_command_spike",
    "quality_has_contact_spike",
)
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
    record_spatial_state: bool = False
    sampling_stride: int = 1
    startup_skip_seconds: float = 0.0
    constraint_mode: str = "static"
    controller: str | None = "sin"
    tau_source: str = "controller_plus_ground"
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
        self._body_indices_by_side: dict[str, list[int]] = {}
        self._body_names_by_side: dict[str, tuple[str, ...]] = {}
        self._joint_dynamics_properties_rows: list[dict[str, Any]] = []
        self._sim_columns: list[str] = []
        self._dynamics_columns: list[str] = []
        self._debug_columns: list[str] = []
        self._spatial_columns: list[str] = []
        self._row_count = 0
        self._tendon_row_count = 0
        self._dynamics_row_count = 0
        self._debug_row_count = 0
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
        self._resolve_joint_dynamics_properties(robot)
        self._create_tables()
        self._initialized = True

    def record_dynamics_step(
        self,
        *,
        step_index: int,
        sim_time: float,
        robot,
        dynamics_terms: dict[str, Any],
        tau_input,
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
                for name in (*DEBUG_DYNAMICS_TERM_NAMES, *DEBUG_DYNAMICS_MATRIX_TERM_NAMES)
                if name not in dynamics_terms
            ]
            if debug_missing:
                raise ValueError(f"Missing debug dynamics term tensors: {debug_missing}")

        for env_id in self._selected_env_ids:
            for side in self._selected_sides():
                joint_indices = self._joint_indices_by_side[side]
                if not self._passes_residual_filter(
                    dynamics_terms=dynamics_terms,
                    env_id=int(env_id),
                    joint_indices=joint_indices,
                ):
                    continue

                if self.cfg.record_dynamics:
                    row_values = [-1, int(step_index), float(sim_time), int(env_id), side]
                    for term_name in TRAINING_DYNAMICS_TERM_NAMES:
                        term_values = self._training_dynamics_term(
                            dynamics_terms=dynamics_terms,
                            term_name=term_name,
                            env_id=int(env_id),
                            joint_indices=joint_indices,
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
                            joint_indices=joint_indices,
                        )
                        debug_values.extend(float(value) for value in term_values)
                    debug_values.extend(
                        self._dynamics_quality_scalars(
                            dynamics_terms=dynamics_terms,
                            env_id=int(env_id),
                            joint_indices=joint_indices,
                        )[name]
                        for name in DEBUG_DYNAMICS_SCALAR_NAMES
                    )
                    for term_name in DEBUG_DYNAMICS_MATRIX_TERM_NAMES:
                        term_tensor = dynamics_terms[term_name][env_id]
                        selected = term_tensor[joint_indices][:, joint_indices].detach().cpu().reshape(-1).tolist()
                        debug_values.extend(float(value) for value in selected)
                    self._debug_rows_by_stream.setdefault((int(env_id), side), []).append(tuple(debug_values))
                    self._debug_row_count += 1

    def _training_dynamics_term(
        self,
        *,
        dynamics_terms: dict[str, Any],
        term_name: str,
        env_id: int,
        joint_indices: list[int],
    ) -> list[float]:
        if term_name == "inertia":
            values = dynamics_terms["inertia"][env_id, joint_indices]
        elif term_name == "coriolis":
            values = -dynamics_terms["coriolis"][env_id, joint_indices]
        elif term_name == "gravity":
            values = -dynamics_terms["gravity"][env_id, joint_indices]
        elif term_name == "tendon":
            values = -dynamics_terms["tendon"][env_id, joint_indices]
        elif term_name == "actuation":
            values = dynamics_terms["actuation_command"][env_id, joint_indices]
        elif term_name == "contact":
            values = dynamics_terms["contact_validated"][env_id, joint_indices]
        elif term_name == "friction":
            values = dynamics_terms["friction"][env_id, joint_indices]
        elif term_name == "external":
            values = (
                dynamics_terms["actuation_command"][env_id, joint_indices]
                + dynamics_terms["contact_validated"][env_id, joint_indices]
                + dynamics_terms["friction"][env_id, joint_indices]
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
        joint_indices: list[int],
    ) -> list[float]:
        if term_name == "inertia_leg_self":
            mass_matrix = dynamics_terms["mass_matrix"][env_id]
            joint_acc = dynamics_terms["joint_acc_for_inertia"][env_id]
            selected_mass = mass_matrix[joint_indices][:, joint_indices]
            selected_acc = joint_acc[joint_indices]
            values = (selected_mass @ selected_acc.unsqueeze(-1)).squeeze(-1)
        elif term_name == "inertia_other_joints":
            mass_matrix = dynamics_terms["mass_matrix"][env_id]
            joint_acc = dynamics_terms["joint_acc_for_inertia"][env_id]
            selected_mass = mass_matrix[joint_indices]
            all_joint = (selected_mass @ joint_acc.unsqueeze(-1)).squeeze(-1)
            leg_self = (selected_mass[:, joint_indices] @ joint_acc[joint_indices].unsqueeze(-1)).squeeze(-1)
            values = all_joint - leg_self
        else:
            values = dynamics_terms[term_name][env_id, joint_indices]
        return values.detach().cpu().tolist()

    def _dynamics_quality_scalars(
        self,
        *,
        dynamics_terms: dict[str, Any],
        env_id: int,
        joint_indices: list[int],
    ) -> dict[str, float]:
        def selected_norm(term_name: str) -> float:
            values = dynamics_terms[term_name][env_id, joint_indices].detach()
            return float((values * values).sum().sqrt().cpu())

        def selected_min(term_name: str) -> float:
            values = dynamics_terms[term_name][env_id, joint_indices].detach()
            return float(values.min().cpu())

        primary_residual = selected_norm("unmodeled")
        sysid_residual = selected_norm("unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal")
        contact_norm = selected_norm("contact_force")
        solver_norm = selected_norm("solver_constraint_internal")
        inertia_norm = selected_norm("inertia")
        command_norm = selected_norm("actuation_command")
        dynamics_scale = max(
            inertia_norm,
            selected_norm("gravity"),
            selected_norm("coriolis"),
            selected_norm("tendon"),
            contact_norm,
            solver_norm,
            command_norm,
            1.0,
        )
        min_limit_distance = selected_min("joint_limit_distance_min")
        max_limit_penetration = max(0.0, -min_limit_distance)
        return {
            "quality_primary_residual_norm": primary_residual,
            "quality_sysid_residual_norm": sysid_residual,
            "quality_sysid_residual_pct": 100.0 * sysid_residual / dynamics_scale,
            "quality_dynamics_scale": dynamics_scale,
            "quality_usable_100": 1.0 if sysid_residual <= 100.0 else 0.0,
            "quality_usable_150": 1.0 if sysid_residual <= 150.0 else 0.0,
            "quality_usable_200": 1.0 if sysid_residual <= 200.0 else 0.0,
            "quality_contact_norm": contact_norm,
            "quality_solver_internal_norm": solver_norm,
            "quality_inertia_norm": inertia_norm,
            "quality_command_norm": command_norm,
            "quality_max_limit_penetration": max_limit_penetration,
            "quality_has_limit_penetration": 1.0 if max_limit_penetration > 0.02 else 0.0,
            "quality_has_inertia_spike": 1.0 if inertia_norm > 1000.0 else 0.0,
            "quality_has_solver_spike": 1.0 if solver_norm > 1000.0 else 0.0,
            "quality_has_command_spike": 1.0 if command_norm > 1000.0 else 0.0,
            "quality_has_contact_spike": 1.0 if contact_norm > 1000.0 else 0.0,
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
        if side not in self._selected_sides():
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
        ddq_all = robot.data.joint_acc if ddq_override is None else ddq_override
        tau_all = self._tau_tensor(robot, tau_override=tau_override)
        context = dict(extra_context or {})

        for env_id in self._selected_env_ids:
            for side in self._selected_sides():
                joint_indices = self._joint_indices_by_side[side]
                if not self._passes_residual_filter(
                    dynamics_terms=dynamics_terms,
                    env_id=int(env_id),
                    joint_indices=joint_indices,
                ):
                    continue
                q = q_all[env_id, joint_indices].detach().cpu().tolist()
                dq = dq_all[env_id, joint_indices].detach().cpu().tolist()
                ddq = ddq_all[env_id, joint_indices].detach().cpu().tolist()
                tau = tau_all[env_id, joint_indices].detach().cpu().tolist()

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
        joint_indices: list[int],
    ) -> bool:
        if self.cfg.residual_filter_threshold is None:
            return True
        if dynamics_terms is None:
            raise RuntimeError("residual_filter_threshold requires dynamics_terms to be provided when recording.")
        residual = dynamics_terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"][
            env_id, joint_indices
        ].detach()
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

    def close(self) -> None:
        """Flush rows, write metadata, and close the SQLite connection."""

        if self._closed:
            return
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

    def _selected_sides(self) -> tuple[str, ...]:
        if self.cfg.side_policy == "left_only":
            return ("left",)
        if self.cfg.side_policy == "right_only":
            return ("right",)
        if self.cfg.side_policy == "both_as_samples":
            return ("left", "right")
        raise NotImplementedError(f"Unsupported side_policy: {self.cfg.side_policy!r}")

    def _joint_names_for_side(self, side: str) -> tuple[str, ...]:
        if self.cfg.selected_joint_names is not None:
            if len(self._selected_sides()) != 1:
                raise ValueError("selected_joint_names can only be used with one selected side.")
            return tuple(self.cfg.selected_joint_names)
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
        return TENDON_CHAIN_LINKS[side]

    def _resolve_joint_indices(self, robot) -> None:
        expected_dofs: int | None = None
        for side in self._selected_sides():
            names = self._joint_names_for_side(side)
            _validate_unique(names, f"{side} selected joint names")
            indices, found_names = robot.find_joints(list(names), preserve_order=True)
            if tuple(found_names) != names:
                raise RuntimeError(f"Could not resolve {side} joints. Requested {names}; found {tuple(found_names)}")
            if expected_dofs is None:
                expected_dofs = len(names)
            elif len(names) != expected_dofs:
                raise RuntimeError("All sides must use the same number of DOFs when stored as separate samples.")
            self._joint_indices_by_side[side] = _to_int_list(indices)
            self._joint_names_by_side[side] = names

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
        sample_id = 0
        for key in self._ordered_stream_keys():
            for row in self._dynamics_rows_by_stream.get(key, ()):
                rows.append((sample_id, *row[1:]))
                sample_id += 1
        return rows

    def _ordered_debug_rows(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = []
        sample_id = 0
        for key in self._ordered_stream_keys():
            for row in self._debug_rows_by_stream.get(key, ()):
                rows.append((sample_id, *row[1:]))
                sample_id += 1
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
            for q_index, (joint_name, joint_index) in enumerate(zip(names, self._joint_indices_by_side[side])):
                rows.append(
                    {
                        "side": side,
                        "q_index": q_index,
                        "joint_name": joint_name,
                        "isaac_joint_index": int(joint_index),
                        "static_friction_coeff": _tensor_scalar(robot.data.joint_friction_coeff, joint_index),
                        "dynamic_friction_coeff": _tensor_scalar(robot.data.joint_dynamic_friction_coeff, joint_index),
                        "viscous_friction_coeff": _tensor_scalar(robot.data.joint_viscous_friction_coeff, joint_index),
                        "armature": _tensor_scalar(robot.data.joint_armature, joint_index),
                    }
                )
        self._joint_dynamics_properties_rows = rows

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
            self._debug_columns = _debug_dynamics_data_columns(self.num_dofs)
            self._debug_db = sqlite3.connect(self.debug_sqlite_path)
            columns_sql = ", ".join(_dynamics_column_sql(name) for name in self._debug_columns)
            self._debug_db.execute(f"CREATE TABLE debug_data ({columns_sql})")
            self._debug_db.execute("CREATE INDEX debug_data_step_idx ON debug_data (step_index, side)")
            self._debug_db.commit()

    def _tau_tensor(self, robot, *, tau_override=None):
        if tau_override is not None:
            return tau_override
        if self.cfg.tau_source == "controller_plus_ground":
            raise RuntimeError("tau_source='controller_plus_ground' requires a tau_override tensor.")
        if self.cfg.tau_source == "applied_torque":
            return robot.data.applied_torque
        if self.cfg.tau_source == "computed_torque":
            return robot.data.computed_torque
        if self.cfg.tau_source == "zero":
            return robot.data.joint_pos * 0.0
        raise ValueError(f"Unsupported tau_source: {self.cfg.tau_source!r}")

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
            "sim_dt": self._sim_dt,
            "selected_env_ids": list(self._selected_env_ids),
            "sample_order": "selected_env_ids in order, then selected_sides in order, then all recorded steps",
            "tau_source": self.cfg.tau_source,
            "residual_filter_threshold": self.cfg.residual_filter_threshold,
            "tau_semantics": self._tau_semantics(),
            "dynamics_semantics": self._dynamics_semantics(),
            "available_tau_sources": ["controller_plus_ground", "applied_torque", "computed_torque", "zero"],
            "sim_units": {"q": "rad", "dq": "rad/s", "ddq": "rad/s^2", "tau": "N*m"},
            "dynamics_units": {"tau": "N*m"},
            "joint_mappings": self._joint_metadata(),
            "joint_dynamics_properties": self._joint_dynamics_properties(),
            "body_mappings": self._body_metadata(),
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
            "selected_sides": list(self._selected_sides()),
            "selected_env_ids": list(self._selected_env_ids),
            "sample_order": "selected_env_ids in order, then selected_sides in order, then all recorded steps",
            "joint_mappings": self._joint_metadata(),
            "body_mappings": self._body_metadata(),
            "config": _jsonable_config(self.cfg),
        }
        self.viz_vars_path.write_text(json.dumps(viz_vars, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _omitted_joint_metadata(self) -> dict[str, list[str]]:
        omitted_by_side = OMITTED_JOINTS_BY_SET.get(self.cfg.joint_set)
        if omitted_by_side is None:
            return {side: [] for side in self._selected_sides()}
        return {side: list(omitted_by_side[side]) for side in self._selected_sides()}

    def _tau_semantics(self) -> str:
        if self.cfg.tau_source == "controller_plus_ground":
            return (
                "actuator generalized torque on actuated joints plus PhysX ground contact generalized torque "
                "from measured contact wrench projected with J^T; uses normal plus friction contact force and "
                "contact-point moment when the contact sensor provides them; tendon forces are intentionally excluded"
            )
        if self.cfg.tau_source == "zero":
            return "zero placeholder for Identix kinematics schema compatibility; do not use as dynamics labels"
        if self.cfg.tau_source == "applied_torque":
            return "IsaacLab robot.data.applied_torque selected by joint index; may not include tendon body wrenches"
        if self.cfg.tau_source == "computed_torque":
            return "IsaacLab robot.data.computed_torque selected by joint index"
        return self.cfg.tau_source

    def _dynamics_semantics(self) -> dict[str, str]:
        if not self.cfg.record_dynamics:
            return {}
        return {
            "sample_id": "zero-based row index aligned one-to-one with sim_data rowid - 1 after optional filtering",
            "equation": (
                "training convention: tau_inertia + tau_coriolis + tau_gravity + tau_tendon = "
                "tau_external, where tau_external = tau_actuation + tau_contact + tau_friction"
            ),
            "tau_inertia": "selected leg rows of the full floating-base generalized inertia product M(q)qdd",
            "tau_coriolis": "Coriolis/centrifugal term on the left-hand side of the training equation",
            "tau_gravity": "gravity term on the left-hand side of the training equation",
            "tau_tendon": "tendon term on the left-hand side of the training equation",
            "tau_actuation": "external actuation command/drive generalized force used for training labels",
            "tau_contact": "external validated contact generalized force projected from measured contact wrench",
            "tau_friction": "external configured joint-friction model term",
            "tau_external": "tau_actuation + tau_contact + tau_friction",
        }
        return {
            "sample_id": "zero-based row index aligned one-to-one with sim_data rowid - 1",
            "tau_inertia": (
                "selected leg rows of the full generalized inertia product using raw IsaacLab/PhysX acceleration "
                "signals. For floating-base robots this includes raw root acceleration coupling plus all-joint "
                "acceleration coupling, then selected back to the recorded 12 leg DOFs."
            ),
            "tau_inertia_recording_interval": (
                "same selected-row full inertia structure as tau_inertia, but using root and joint accelerations "
                "finite-differenced over the recorder's row interval. Kept as a diagnostic because this estimate has "
                "shown worse residual tails than raw PhysX/IsaacLab acceleration signals."
            ),
            "tau_inertia_raw": (
                "selected rows of PhysX generalized mass matrix multiplied by IsaacLab robot.data.joint_acc. "
                "In decimated RL loops this is a last-physics-substep acceleration diagnostic, not necessarily the "
                "average acceleration over the recorded row interval."
            ),
            "tau_inertia_joint_only": (
                "legacy alias of tau_inertia_joint_all: joint block of the generalized mass matrix multiplied by all "
                "joint accelerations, without floating-root acceleration coupling"
            ),
            "tau_inertia_joint_all": (
                "joint-row generalized inertia contribution from all joint accelerations: M_jj * qdd_all_joints"
            ),
            "tau_inertia_leg_self": (
                "selected leg self-coupling contribution computed per recorded side: M_leg,leg * qdd_leg"
            ),
            "tau_inertia_other_joints": (
                "joint acceleration coupling from joints outside the recorded side: M_leg,other * qdd_other"
            ),
            "tau_inertia_root_coupling": (
                "joint-row generalized inertia contribution from finite-differenced floating root velocity over the "
                "recording interval when available, otherwise PhysX link acceleration order [linear, angular]"
            ),
            "tau_inertia_root_coupling_raw": (
                "joint-row generalized inertia contribution from PhysX link acceleration order [linear, angular] "
                "at the sample instant"
            ),
            "tau_inertia_root_coupling_alt": (
                "same root coupling diagnostic but with root acceleration order [angular, linear]"
            ),
            "tau_inertia_root_coupled_alt": (
                "tau_inertia_joint_only plus tau_inertia_root_coupling_alt; used to test root coordinate ordering"
            ),
            "tau_inertia_full_raw": (
                "tau_inertia_raw plus tau_inertia_root_coupling_raw, using raw IsaacLab/PhysX acceleration signals"
            ),
            "tau_coriolis": (
                "PhysX actual generalized Coriolis and centrifugal forces for the current articulation state"
            ),
            "tau_gravity": "PhysX actual generalized gravity forces for the current articulation pose",
            "tau_friction_dynamic": "dynamic Coulomb joint-friction model term: -dynamic_friction_coeff * sign(dq)",
            "tau_friction_viscous": "viscous joint-friction model term: -viscous_friction_coeff * dq",
            "tau_friction": (
                "model estimate from configured joint dynamic and viscous friction coefficients; static friction is "
                "stored in metadata because its active solver value is not exposed as a separated generalized force"
            ),
            "tau_solver_joint": (
                "PhysX DOF projected joint forces: the active component obtained by projecting each link incoming "
                "joint force onto the joint motion direction; this is the closest solver-measured joint-space force "
                "signal exposed by the tensor API, not a decomposition by source"
            ),
            "tau_actuation": (
                "PhysX-measured DOF actuation force used in the primary force balance. This is currently the same "
                "source as tau_physx_actuation and may be zero for implicit actuators because PhysX does not expose "
                "their solved drive force through this tensor API."
            ),
            "tau_actuation_command": (
                "IsaacLab applied_torque estimate aligned to Isaac joint indices. For implicit actuators this is an "
                "approximate command/drive estimate, not a measured physical generalized force, and is not subtracted "
                "in the primary tau_unmodeled balance."
            ),
            "tau_actuation_estimated": (
                "estimated implicit drive generalized force from IsaacLab applied_torque. This is recorded as a "
                "diagnostic/model candidate because PhysX does not expose the solved implicit drive force as a "
                "measured tensor force."
            ),
            "tau_actuation_estimated_hip": (
                "estimated implicit drive generalized force only on hip roll, hip lateral, and hip flexion joints. "
                "Pantograph and knee-flexor estimated drive are excluded because they have not matched the residual "
                "balance in Forrest diagnostics."
            ),
            "tau_actuation_estimated_hip_lateral_flexion": (
                "estimated implicit drive generalized force only on hip lateral and hip flexion joints. Hip roll is "
                "excluded because its residual is dominated by limit/solver reaction diagnostics in recent runs."
            ),
            "tau_actuation_estimated_passive": (
                "estimated implicit drive generalized force on passive tendon-chain joints only. This is recorded to "
                "test whether configured implicit drives or constraints are injecting force on joints that should be "
                "modeled by tendon/contact terms."
            ),
            "tau_physx_actuation": "PhysX DOF actuation forces reported by the articulation tensor API",
            "tau_solver_constraint_passive": (
                "selected PhysX solver-projected joint force on passive-chain joints whose residuals previously "
                "matched solver reactions. This is a diagnostic proxy for internal joint/constraint reactions, not a "
                "source-separated applied force."
            ),
            "tau_solver_constraint_limit": (
                "PhysX solver-projected joint force recorded only on DOFs whose soft-limit distance is <= 0.05 rad. "
                "This is the best available diagnostic for joint-limit inner-contact reactions."
            ),
            "tau_solver_constraint_internal": (
                "union of tau_solver_constraint_passive and tau_solver_constraint_limit masks, using the raw "
                "solver-projected force wherever either diagnostic mask is active"
            ),
            "tau_joint_drive_pos_target": "IsaacLab joint position target recorded to audit implicit-drive state",
            "tau_joint_drive_vel_target": "IsaacLab joint velocity target recorded to audit implicit-drive state",
            "tau_joint_drive_effort_target": "IsaacLab joint effort target recorded to audit implicit-drive state",
            "tau_joint_drive_stiffness": "IsaacLab joint drive stiffness at the sample instant",
            "tau_joint_drive_damping": "IsaacLab joint drive damping at the sample instant",
            "tau_joint_effort_limit": "IsaacLab/PhysX joint effort limit at the sample instant",
            "tau_joint_velocity_limit": "IsaacLab soft joint velocity limit at the sample instant",
            "tau_joint_limit_lower": "IsaacLab soft lower joint-position limit",
            "tau_joint_limit_upper": "IsaacLab soft upper joint-position limit",
            "tau_joint_limit_distance_lower": "joint_pos minus soft lower joint limit",
            "tau_joint_limit_distance_upper": "soft upper joint limit minus joint_pos",
            "tau_joint_limit_distance_min": "minimum signed distance to either soft joint-position limit",
            "tau_drive_stiffness": (
                "implicit-drive stiffness estimate: joint_stiffness * (joint_pos_target - joint_pos). PhysX solves "
                "implicit drives internally, so this is a model estimate, not a measured solver force."
            ),
            "tau_drive_damping": (
                "implicit-drive damping estimate: joint_damping * (joint_vel_target - joint_vel). This is separate "
                "from tau_friction_viscous, which uses the configured joint viscous friction coefficient."
            ),
            "tau_drive_effort_target": "feed-forward joint effort target sent to the implicit drive",
            "tau_drive_pd": (
                "unclipped implicit-drive PD estimate: drive_stiffness + drive_damping + drive_effort_target"
            ),
            "tau_drive_pd_clipped": "tau_drive_pd clipped to the configured joint effort limits",
            "tau_armature_inertia": (
                "joint armature times raw joint acceleration. The generalized mass matrix should already include "
                "armature; this is recorded only to detect double-counting or missing-armature hypotheses."
            ),
            "tau_contact": (
                "raw full measured contact projection for tracked contact bodies: tau_contact_force plus "
                "tau_contact_moment. This is kept for auditing and may contain invalid contact-point moment outliers."
            ),
            "tau_contact_force": "linear contact force contribution only, projected with the linear Jacobian block",
            "tau_contact_moment": (
                "contact moment contribution only, computed from estimated contact point relative to body origin and "
                "projected with the angular Jacobian block"
            ),
            "tau_contact_validated": (
                "contact force plus contact moment only when the projected generalized moment/force ratio "
                "|tau_contact_moment| / |tau_contact_force| is <= 2; otherwise force-only contact is used"
            ),
            "tau_contact_digit": "contact projection restricted to digit/toe contact bodies",
            "tau_contact_digit_force": "linear-force part of tau_contact_digit",
            "tau_contact_digit_moment": "contact-point moment part of tau_contact_digit",
            "tau_contact_connector": "contact projection restricted to foot connector bodies",
            "tau_contact_connector_force": "linear-force part of tau_contact_connector",
            "tau_contact_connector_moment": "contact-point moment part of tau_contact_connector",
            "tau_contact_base": "contact projection restricted to base, hip, and differential-cage bodies",
            "tau_contact_base_force": "linear-force part of tau_contact_base",
            "tau_contact_base_moment": "contact-point moment part of tau_contact_base",
            "tau_tendon": (
                "cached tendon body wrenches projected into joint space with J^T. This is the applied generalized "
                "force from the wrench actually sent to PhysX; if the wrench cache is unavailable, recording falls "
                "back to the older model joint-torque cache."
            ),
            "tau_tendon_model": "raw tendon model joint-torque cache before mapping to link/body wrenches",
            "tau_tendon_projection_delta": "tau_tendon minus tau_tendon_model, useful for validating wrench projection",
            "mass_matrix": (
                "selected DOF-by-DOF PhysX generalized mass matrix used to recompute filtered inertia offline"
            ),
            "tau_unmodeled_full_contact": (
                "residual using raw full contact: tau_inertia - tau_gravity - tau_coriolis - tau_tendon "
                "- tau_actuation - tau_contact - tau_friction"
            ),
            "tau_unmodeled_contact_force_only": (
                "residual using contact force only: tau_inertia - tau_gravity - tau_coriolis - tau_tendon "
                "- tau_actuation - tau_contact_force - tau_friction"
            ),
            "tau_unmodeled_contact_validated": (
                "residual using tau_contact_validated; this is the preferred residual while contact-point moments "
                "are under validation"
            ),
            "tau_unmodeled_quasistatic": (
                "residual with the inertia term removed: -gravity - coriolis - tendon - actuation "
                "- contact_validated - friction. This is recorded because inertia signals are still being validated."
            ),
            "tau_unmodeled_full_dynamics": (
                "full inverse-dynamics residual using tau_inertia; equivalent to tau_unmodeled_contact_validated"
            ),
            "tau_unmodeled_recording_interval": (
                "same full inverse-dynamics residual as tau_unmodeled, but using tau_inertia_recording_interval"
            ),
            "tau_unmodeled_estimated_actuation": (
                "full inverse-dynamics residual using tau_actuation_estimated instead of measured tau_actuation"
            ),
            "tau_unmodeled_estimated_hip_actuation": (
                "full inverse-dynamics residual using tau_actuation_estimated_hip instead of measured tau_actuation"
            ),
            "tau_unmodeled_estimated_hip_force_contact": (
                "full inverse-dynamics residual using hip-only estimated actuation and contact_force instead of "
                "measured actuation and contact_validated. This tests the best current measured/contact hypothesis."
            ),
            "tau_unmodeled_estimated_hip_force_contact_solver": (
                "diagnostic residual using hip-only estimated actuation, contact_force, and selected passive solver "
                "constraint proxy. This is not the primary physical residual, but it helps identify missing internal "
                "constraint/limit forces."
            ),
            "tau_unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal": (
                "diagnostic residual using hip lateral/flexion estimated actuation, contact_force, and the combined "
                "passive-or-limit solver proxy. This tests whether hip-roll should be treated as an internal "
                "limit/solver reaction instead of a clean actuator force."
            ),
            "tau_unmodeled": (
                "alias of tau_unmodeled_contact_validated. Uses recorded actual-force signs: "
                "tau_inertia - tau_gravity - tau_coriolis - tau_tendon - tau_actuation - tau_contact_validated "
                "- tau_friction. "
                "Gravity, Coriolis/centrifugal, and tendon are treated as potential/bias effects on the required side; "
                "actuation, contact, and friction are treated as applied generalized forces."
            ),
            "tau_inverse_residual": ("alias of tau_unmodeled for the primary inverse-dynamics balance"),
            "tau_solver_residual": (
                "solver-projection comparison: tau_solver_joint minus actuation, contact, and friction. "
                "tau_solver_joint is a projected joint-reaction diagnostic, not a total generalized force."
            ),
            "quality_*": (
                "row-level sysid filtering metrics computed on the recorded side's selected 12 DOFs. "
                "quality_sysid_residual_norm uses the current best diagnostic residual, while tau_unmodeled remains "
                "the primary physical balance residual."
            ),
        }

    def _report_dynamics_residual(self) -> None:
        if not self.cfg.record_debug_dynamics or self._debug_db is None or self._debug_row_count == 0:
            return

        term_names = (
            "inertia",
            "inertia_recording_interval",
            "inertia_raw",
            "inertia_joint_only",
            "inertia_joint_all",
            "inertia_leg_self",
            "inertia_other_joints",
            "inertia_root_coupling",
            "inertia_root_coupling_raw",
            "inertia_root_coupling_alt",
            "inertia_root_coupled_alt",
            "inertia_full_raw",
            "coriolis",
            "gravity",
            "friction_dynamic",
            "friction_viscous",
            "friction",
            "solver_joint",
            "actuation",
            "actuation_command",
            "actuation_estimated",
            "actuation_estimated_hip",
            "actuation_estimated_hip_lateral_flexion",
            "actuation_estimated_passive",
            "physx_actuation",
            "solver_constraint_passive",
            "solver_constraint_limit",
            "solver_constraint_internal",
            "joint_drive_pos_target",
            "joint_drive_vel_target",
            "joint_drive_effort_target",
            "joint_drive_stiffness",
            "joint_drive_damping",
            "joint_effort_limit",
            "joint_velocity_limit",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_distance_lower",
            "joint_limit_distance_upper",
            "joint_limit_distance_min",
            "drive_stiffness",
            "drive_damping",
            "drive_effort_target",
            "drive_pd",
            "drive_pd_clipped",
            "armature_inertia",
            "contact",
            "contact_force",
            "contact_moment",
            "contact_validated",
            "contact_digit",
            "contact_digit_force",
            "contact_digit_moment",
            "contact_connector",
            "contact_connector_force",
            "contact_connector_moment",
            "contact_base",
            "contact_base_force",
            "contact_base_moment",
            "tendon",
            "tendon_model",
            "tendon_projection_delta",
            "unmodeled_quasistatic",
            "unmodeled_full_dynamics",
            "unmodeled_recording_interval",
            "unmodeled_estimated_actuation",
            "unmodeled_estimated_hip_actuation",
            "unmodeled_estimated_hip_force_contact",
            "unmodeled_estimated_hip_force_contact_solver",
            "unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal",
            "unmodeled_full_contact",
            "unmodeled_contact_force_only",
            "unmodeled_contact_validated",
            "unmodeled",
            "inverse_residual",
            "solver_residual",
        )
        matrix_term_names = DEBUG_DYNAMICS_MATRIX_TERM_NAMES
        term_cols = {name: [f"tau_{name}{i}" for i in range(self.num_dofs)] for name in term_names}
        matrix_cols = {
            name: [f"{name}{row}_{col}" for row in range(self.num_dofs) for col in range(self.num_dofs)]
            for name in matrix_term_names
        }
        selected_cols = [
            "sample_id",
            "step_index",
            "time",
            "env_id",
            "side",
            *(column for name in term_names for column in term_cols[name]),
            *(column for name in matrix_term_names for column in matrix_cols[name]),
        ]
        quoted_cols = ", ".join(_quote_identifier(name) for name in selected_cols)
        rows = self._debug_db.execute(f"SELECT {quoted_cols} FROM debug_data ORDER BY sample_id").fetchall()
        if not rows:
            return

        def vector_norm(values: tuple[float, ...]) -> float:
            return math.sqrt(sum(value * value for value in values))

        def add(*vectors: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(sum(vector[i] for vector in vectors) for i in range(self.num_dofs))

        def neg(vector: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(-value for value in vector)

        def unpack_terms(row: tuple[Any, ...]) -> dict[str, tuple[float, ...]]:
            offset = 5
            terms = {"_sample_id": int(row[0])}
            for name in term_names:
                terms[name] = tuple(float(value) for value in row[offset : offset + self.num_dofs])
                offset += self.num_dofs
            for name in matrix_term_names:
                values = tuple(float(value) for value in row[offset : offset + self.num_dofs * self.num_dofs])
                terms[name] = tuple(values[i * self.num_dofs : (i + 1) * self.num_dofs] for i in range(self.num_dofs))
                offset += self.num_dofs * self.num_dofs
            return terms

        def primary_required(terms: dict[str, tuple[float, ...]]) -> tuple[float, ...]:
            return add(terms["inertia"], neg(terms["gravity"]), neg(terms["coriolis"]), neg(terms["tendon"]))

        def primary_applied(terms: dict[str, tuple[float, ...]]) -> tuple[float, ...]:
            return add(terms["actuation"], terms["contact_validated"], terms["friction"])

        def dynamics_scale(terms: dict[str, tuple[float, ...]], residual: tuple[float, ...]) -> float:
            return max(
                vector_norm(primary_required(terms)),
                vector_norm(primary_applied(terms)),
                vector_norm(terms["solver_joint"]),
                1.0e-9,
            )

        def term_norm_summary(name: str) -> tuple[float, float]:
            norms = [vector_norm(terms[name]) for _row, terms in unpacked_rows]
            return sum(norms) / len(norms), max(norms)

        candidates = (
            (
                "validated",
                "Mqdd - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: terms["unmodeled"],
            ),
            (
                "record interval",
                "Mqdd_record_interval - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: terms["unmodeled_recording_interval"],
            ),
            (
                "quasistatic",
                "-gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: terms["unmodeled_quasistatic"],
            ),
            (
                "full contact",
                "Mqdd - gravity - coriolis - tendon - actuation - contact - friction",
                lambda terms: terms["unmodeled_full_contact"],
            ),
            (
                "contact force only",
                "Mqdd - gravity - coriolis - tendon - actuation - contact_force - friction",
                lambda terms: terms["unmodeled_contact_force_only"],
            ),
            (
                "flip tendon",
                "Mqdd - gravity - coriolis + tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    terms["tendon"],
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "flip contact",
                "Mqdd - gravity - coriolis - tendon - actuation + contact_validated - friction",
                lambda terms: add(
                    terms["inertia"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    terms["contact_validated"],
                    neg(terms["friction"]),
                ),
            ),
            (
                "flip actuation",
                "Mqdd - gravity - coriolis - tendon + actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    terms["actuation"],
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "use command",
                "Mqdd - gravity - coriolis - tendon - actuation_command - contact_validated - friction",
                lambda terms: add(
                    terms["inertia"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation_command"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "estimated actuation",
                "Mqdd - gravity - coriolis - tendon - actuation_estimated - contact_validated - friction",
                lambda terms: terms["unmodeled_estimated_actuation"],
            ),
            (
                "estimated hip act",
                "Mqdd - gravity - coriolis - tendon - actuation_estimated_hip - contact_validated - friction",
                lambda terms: terms["unmodeled_estimated_hip_actuation"],
            ),
            (
                "hip act force contact",
                ("Mqdd - gravity - coriolis - tendon - actuation_estimated_hip - contact_force - friction"),
                lambda terms: terms["unmodeled_estimated_hip_force_contact"],
            ),
            (
                "hip+solver diag",
                (
                    "Mqdd - gravity - coriolis - tendon - "
                    "actuation_estimated_hip - contact_force - friction - "
                    "solver_constraint_passive"
                ),
                lambda terms: terms["unmodeled_estimated_hip_force_contact_solver"],
            ),
            (
                "hip23+internal",
                (
                    "Mqdd - gravity - coriolis - tendon - "
                    "actuation_estimated_hip_lateral_flexion - contact_force - "
                    "friction - solver_constraint_internal"
                ),
                lambda terms: terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"],
            ),
            (
                "drive pd clipped",
                (
                    "Mqdd - gravity - coriolis - tendon - drive_pd_clipped - "
                    "contact_force - friction - solver_constraint_internal"
                ),
                lambda terms: add(
                    terms["inertia"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["drive_pd_clipped"]),
                    neg(terms["contact_force"]),
                    neg(terms["friction"]),
                    neg(terms["solver_constraint_internal"]),
                ),
            ),
            (
                "hip23+int no fric",
                (
                    "Mqdd - gravity - coriolis - tendon - "
                    "actuation_estimated_hip_lateral_flexion - contact_force - "
                    "solver_constraint_internal"
                ),
                lambda terms: add(
                    terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"],
                    terms["friction"],
                ),
            ),
            (
                "hip23+int no arm",
                (
                    "Mqdd_minus_armature - gravity - coriolis - tendon - "
                    "actuation_estimated_hip_lateral_flexion - contact_force - "
                    "friction - solver_constraint_internal"
                ),
                lambda terms: add(
                    terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"],
                    neg(terms["armature_inertia"]),
                ),
            ),
            (
                "comp sign",
                "Mqdd + gravity + coriolis + tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia"],
                    terms["gravity"],
                    terms["coriolis"],
                    terms["tendon"],
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "solver diag",
                "solver_projection - actuation - contact_validated - friction",
                lambda terms: terms["solver_residual"],
            ),
            (
                "solver+command",
                "solver_projection - actuation_command - contact_validated - friction",
                lambda terms: add(
                    terms["solver_joint"],
                    neg(terms["actuation_command"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "tendon model",
                "Mqdd - gravity - coriolis - tendon_model - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon_model"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "all-joint inertia",
                "Mqdd_all_joints - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia_joint_all"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "leg self inertia",
                "Mqdd_leg_self - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia_leg_self"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "base+self inertia",
                (
                    "Mqdd_base_coupling + Mqdd_leg_self - gravity - coriolis - "
                    "tendon - actuation - contact_validated - friction"
                ),
                lambda terms: add(
                    terms["inertia_root_coupling"],
                    terms["inertia_leg_self"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "base only inertia",
                "Mqdd_base_coupling - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia_root_coupling"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "raw full inertia",
                "Mqdd_full_raw - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia_full_raw"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "raw substep inertia",
                "Mqdd_raw - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia_raw"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "alt root inertia",
                "Mqdd_root_order_alt - gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    terms["inertia_root_coupled_alt"],
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
            (
                "no inertia",
                "-gravity - coriolis - tendon - actuation - contact_validated - friction",
                lambda terms: add(
                    neg(terms["gravity"]),
                    neg(terms["coriolis"]),
                    neg(terms["tendon"]),
                    neg(terms["actuation"]),
                    neg(terms["contact_validated"]),
                    neg(terms["friction"]),
                ),
            ),
        )

        rng = random.Random(17)
        samples: list[tuple[tuple[Any, ...], dict[str, tuple[float, ...]], tuple[float, ...], float]] = []
        candidate_summaries = []
        primary_worst = None
        eps = 1.0e-9

        unpacked_rows = [(row, unpack_terms(row)) for row in rows]
        filtered_inertia_by_sample = self._filtered_inertia_by_sample(unpacked_rows)
        if filtered_inertia_by_sample:
            candidates = (
                *candidates,
                (
                    "central inertia",
                    "M*central_diff(dq) - gravity - coriolis - tendon - actuation - contact_validated - friction",
                    lambda terms: add(
                        filtered_inertia_by_sample.get(int(terms["_sample_id"]), terms["inertia"]),
                        neg(terms["gravity"]),
                        neg(terms["coriolis"]),
                        neg(terms["tendon"]),
                        neg(terms["actuation"]),
                        neg(terms["contact_validated"]),
                        neg(terms["friction"]),
                    ),
                ),
            )
        for candidate_index, (name, equation, residual_fn) in enumerate(candidates):
            percentages: list[float] = []
            residual_norms: list[float] = []
            worst = None
            worst_percentage = -1.0
            max_abs_residual = 0.0
            for index, (row, terms) in enumerate(unpacked_rows):
                residual = residual_fn(terms)
                scale = dynamics_scale(terms, residual)
                residual_norm = vector_norm(residual)
                percentage = 100.0 * residual_norm / max(scale, eps)
                percentages.append(percentage)
                residual_norms.append(residual_norm)
                max_abs_residual = max(max_abs_residual, max(abs(value) for value in residual))
                if percentage > worst_percentage:
                    worst_percentage = percentage
                    worst = (row, terms, residual, percentage)
                if candidate_index == 0:
                    if len(samples) < 5:
                        samples.append((row, terms, residual, percentage))
                    else:
                        replace_index = rng.randint(0, index)
                        if replace_index < len(samples):
                            samples[replace_index] = (row, terms, residual, percentage)

            percentages_sorted = sorted(percentages)
            residual_norms_sorted = sorted(residual_norms)
            p95 = percentages_sorted[min(len(percentages_sorted) - 1, int(0.95 * (len(percentages_sorted) - 1)))]
            mean = sum(percentages) / len(percentages)
            p95_norm = residual_norms_sorted[
                min(len(residual_norms_sorted) - 1, int(0.95 * (len(residual_norms_sorted) - 1)))
            ]
            candidate_summaries.append(
                {
                    "name": name,
                    "equation": equation,
                    "mean_percent_of_dynamics_scale": mean,
                    "p95_percent_of_dynamics_scale": p95,
                    "max_percent_of_dynamics_scale": worst_percentage,
                    "mean_residual_norm_nm": sum(residual_norms) / len(residual_norms),
                    "p95_residual_norm_nm": p95_norm,
                    "max_residual_norm_nm": max(residual_norms),
                    "max_abs_residual_nm": max_abs_residual,
                }
            )
            if candidate_index == 0:
                primary_worst = worst

        primary_summary = candidate_summaries[0]
        ranked_summaries = sorted(candidate_summaries, key=lambda item: item["mean_percent_of_dynamics_scale"])
        self._context_metadata["dynamics_residual_summary"] = {
            "rows": len(rows),
            "primary": primary_summary,
            "ranked_hypotheses": ranked_summaries,
        }

        print("\n[ForrestDynamics] DOF unmodeled-force check")
        print("  Primary equation:")
        print("    tau_unmodeled = Mqdd - gravity - coriolis - tendon - actuation - contact_validated - friction")
        print("    Mqdd is the selected leg rows of the full floating-base inertia: base + leg self + other joints")
        print("    primary Mqdd uses raw IsaacLab/PhysX root and joint acceleration signals")
        print("    recording-interval finite-difference inertia is recorded separately for comparison")
        print("    contact_validated = contact_force + contact_moment only when |moment| / |force| <= 2")
        print("    raw full-contact and force-only residuals are recorded separately for comparison")
        print("    gravity/coriolis/tendon are recorded as actual generalized forces, not compensation commands")
        print("    actuation/contact/friction are measured or estimated applied generalized forces")
        print("    actuation_command is recorded separately and is not subtracted in the primary balance")
        print("    actuation_estimated is an IsaacLab implicit-drive estimate and is tested as a diagnostic candidate")
        print("    actuation_estimated_hip keeps only hip roll/lateral/flexion estimated drive")
        print("    actuation_estimated_hip_lateral_flexion excludes hip roll to test q1 limit/solver behavior")
        print("    solver_constraint_passive is a selected passive-chain solver-force diagnostic, not primary physics")
        print("    solver_constraint_limit records solver-projected force where soft-limit distance <= 0.05 rad")
        print(f"  rows: {len(rows):,}")
        print("  signal norm diagnostics (mean/max N*m):")
        for name in (
            "inertia",
            "inertia_recording_interval",
            "inertia_raw",
            "inertia_joint_only",
            "inertia_joint_all",
            "inertia_leg_self",
            "inertia_other_joints",
            "inertia_root_coupling",
            "inertia_root_coupling_raw",
            "inertia_root_coupling_alt",
            "inertia_root_coupled_alt",
            "inertia_full_raw",
            "armature_inertia",
            "actuation",
            "actuation_command",
            "actuation_estimated",
            "actuation_estimated_hip",
            "actuation_estimated_hip_lateral_flexion",
            "actuation_estimated_passive",
            "solver_constraint_passive",
            "solver_constraint_limit",
            "solver_constraint_internal",
            "drive_stiffness",
            "drive_damping",
            "drive_effort_target",
            "drive_pd",
            "drive_pd_clipped",
            "contact",
            "contact_force",
            "contact_moment",
            "contact_validated",
            "contact_digit",
            "contact_digit_force",
            "contact_digit_moment",
            "contact_connector",
            "contact_connector_force",
            "contact_connector_moment",
            "contact_base",
            "contact_base_force",
            "contact_base_moment",
            "tendon",
            "tendon_model",
            "tendon_projection_delta",
            "unmodeled_quasistatic",
            "unmodeled_full_dynamics",
            "unmodeled_recording_interval",
            "unmodeled_estimated_actuation",
            "unmodeled_estimated_hip_actuation",
            "unmodeled_estimated_hip_force_contact",
            "unmodeled_estimated_hip_force_contact_solver",
            "unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal",
            "unmodeled_full_contact",
            "unmodeled_contact_force_only",
            "unmodeled_contact_validated",
        ):
            mean_norm, max_norm = term_norm_summary(name)
            print(f"    {name:<18} {mean_norm:8.3f} / {max_norm:8.3f}")
        self._print_dynamics_quality_gates(unpacked_rows)
        self._print_dynamics_forensics(unpacked_rows, filtered_inertia_by_sample)
        print(
            "  primary residual / dynamics scale: "
            f"mean={primary_summary['mean_percent_of_dynamics_scale']:6.2f}%  "
            f"p95={primary_summary['p95_percent_of_dynamics_scale']:6.2f}%  "
            f"max={primary_summary['max_percent_of_dynamics_scale']:6.2f}%"
        )
        print(f"  primary max |tau_unmodeled_i|: {primary_summary['max_abs_residual_nm']:8.3f} N*m")
        print("  sign-hypothesis ranking by mean residual:")
        for summary in ranked_summaries:
            print(
                f"    {summary['name']:<14} "
                f"mean={summary['mean_percent_of_dynamics_scale']:6.2f}%  "
                f"p95={summary['p95_percent_of_dynamics_scale']:6.2f}%  "
                f"max={summary['max_percent_of_dynamics_scale']:6.2f}%  "
                f"mean|r|={summary['mean_residual_norm_nm']:8.3f} N*m  "
                f"p95|r|={summary['p95_residual_norm_nm']:8.3f} N*m  "
                f"max|tau|={summary['max_abs_residual_nm']:8.3f} N*m"
            )
        print("  random samples:")
        for row, _terms, residual, percentage in samples:
            self._print_residual_sample(row, residual, percentage)
        if primary_worst is not None:
            print("  worst sample:")
            row, _terms, residual, percentage = primary_worst
            self._print_residual_sample(row, residual, percentage)
        print()

    def _print_residual_sample(self, row: tuple[Any, ...], residual: tuple[float, ...], percentage: float) -> None:
        max_index = max(range(self.num_dofs), key=lambda i: abs(float(residual[i]))) if self.num_dofs > 0 else 0
        joint_names = self._joint_names_by_side.get(str(row[4]), ())
        joint_label = joint_names[max_index] if max_index < len(joint_names) else f"q{max_index}"
        print(
            "    "
            f"sample={int(row[0]):>7} step={int(row[1]):>6} env={int(row[3]):>4} side={row[4]:>5} "
            f"residual={percentage:6.2f}% max={float(residual[max_index]):+8.3f} N*m "
            f"at q{max_index} ({joint_label})"
        )

    def _filtered_inertia_by_sample(
        self, unpacked_rows: list[tuple[tuple[Any, ...], dict[str, Any]]]
    ) -> dict[int, tuple[float, ...]]:
        if self._db is None or not unpacked_rows:
            return {}

        dq_cols = [f"dq{i}" for i in range(self.num_dofs)]
        quoted = ", ".join(_quote_identifier(name) for name in dq_cols)
        sim_rows = self._db.execute(f"SELECT rowid, {quoted} FROM {self.cfg.sim_table_name} ORDER BY rowid").fetchall()
        if len(sim_rows) < 3:
            return {}

        times = [float(row[2]) for row, _terms in unpacked_rows]
        dq_by_sample = {int(rowid) - 1: tuple(float(value) for value in values) for rowid, *values in sim_rows}
        filtered: dict[int, tuple[float, ...]] = {}
        for index, (row, terms) in enumerate(unpacked_rows):
            sample_id = int(row[0])
            if sample_id not in dq_by_sample:
                continue
            prev_same_stream = (
                index > 0
                and int(unpacked_rows[index - 1][0][3]) == int(row[3])
                and str(unpacked_rows[index - 1][0][4]) == str(row[4])
            )
            next_same_stream = (
                index + 1 < len(unpacked_rows)
                and int(unpacked_rows[index + 1][0][3]) == int(row[3])
                and str(unpacked_rows[index + 1][0][4]) == str(row[4])
            )
            if prev_same_stream and next_same_stream:
                prev_id, next_id = int(unpacked_rows[index - 1][0][0]), int(unpacked_rows[index + 1][0][0])
                dt = max(times[index + 1] - times[index - 1], 1.0e-9)
            elif next_same_stream:
                prev_id, next_id = sample_id, int(unpacked_rows[index + 1][0][0])
                dt = max(times[index + 1] - times[index], 1.0e-9)
            elif prev_same_stream:
                prev_id, next_id = int(unpacked_rows[index - 1][0][0]), sample_id
                dt = max(times[index] - times[index - 1], 1.0e-9)
            else:
                continue
            if prev_id not in dq_by_sample or next_id not in dq_by_sample:
                continue
            ddq = tuple((dq_by_sample[next_id][i] - dq_by_sample[prev_id][i]) / dt for i in range(self.num_dofs))
            mass_matrix = terms["mass_matrix"]
            filtered[sample_id] = tuple(
                sum(float(mass_matrix[i][j]) * ddq[j] for j in range(self.num_dofs)) for i in range(self.num_dofs)
            )
        return filtered

    def _print_dynamics_forensics(
        self,
        unpacked_rows: list[tuple[tuple[Any, ...], dict[str, Any]]],
        filtered_inertia_by_sample: dict[int, tuple[float, ...]],
    ) -> None:
        if not unpacked_rows:
            return

        def vector_norm(values: tuple[float, ...]) -> float:
            return math.sqrt(sum(value * value for value in values))

        def flatten(name: str) -> list[float]:
            return [float(value) for _row, terms in unpacked_rows for value in terms[name]]

        def corr(left: list[float], right: list[float]) -> float:
            if len(left) != len(right) or not left:
                return float("nan")
            mean_left = sum(left) / len(left)
            mean_right = sum(right) / len(right)
            var_left = sum((value - mean_left) ** 2 for value in left)
            var_right = sum((value - mean_right) ** 2 for value in right)
            if var_left <= 1.0e-12 or var_right <= 1.0e-12:
                return float("nan")
            cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
            return cov / math.sqrt(var_left * var_right)

        residual = flatten("unmodeled")
        print("  forensic correlations with tau_unmodeled:")
        for name in (
            "inertia",
            "inertia_recording_interval",
            "inertia_raw",
            "inertia_joint_only",
            "inertia_joint_all",
            "inertia_leg_self",
            "inertia_other_joints",
            "inertia_root_coupling",
            "inertia_root_coupling_raw",
            "inertia_root_coupling_alt",
            "inertia_root_coupled_alt",
            "inertia_full_raw",
            "contact",
            "contact_force",
            "contact_moment",
            "contact_validated",
            "contact_digit",
            "contact_connector",
            "contact_base",
            "tendon",
            "tendon_model",
            "actuation_command",
            "actuation_estimated",
            "actuation_estimated_hip",
            "actuation_estimated_hip_lateral_flexion",
            "actuation_estimated_passive",
            "solver_constraint_passive",
            "solver_constraint_limit",
            "solver_constraint_internal",
            "drive_stiffness",
            "drive_damping",
            "drive_pd_clipped",
            "friction_dynamic",
            "friction_viscous",
            "armature_inertia",
        ):
            print(f"    {name:<18} corr={corr(residual, flatten(name)):+6.3f}")

        invalid_contact_rows = []
        for row, terms in unpacked_rows:
            contact_force_norm = vector_norm(terms["contact_force"])
            contact_moment_norm = vector_norm(terms["contact_moment"])
            projected_moment_ratio = contact_moment_norm / max(contact_force_norm, 1.0e-9)
            if projected_moment_ratio > 2.0:
                invalid_contact_rows.append(
                    (projected_moment_ratio, contact_force_norm, contact_moment_norm, row, terms)
                )
        if invalid_contact_rows:
            invalid_contact_rows.sort(key=lambda item: item[0], reverse=True)
            print("  invalid contact-moment rows (projected |moment| / |force| > 2):")
            print(
                "    "
                f"count={len(invalid_contact_rows):,} / {len(unpacked_rows):,}  "
                f"max_ratio={invalid_contact_rows[0][0]:.2f}"
            )
            for projected_moment_ratio, contact_force_norm, contact_moment_norm, row, terms in invalid_contact_rows[:5]:
                max_index = max(range(self.num_dofs), key=lambda i: abs(float(terms["contact_moment"][i])))
                print(
                    "    "
                    f"sample={int(row[0]):>7} step={int(row[1]):>6} time={float(row[2]):7.3f}s "
                    f"ratio={projected_moment_ratio:6.2f}  "
                    f"|force|={contact_force_norm:8.3f} N*m  "
                    f"|moment|={contact_moment_norm:8.3f} N*m  "
                    f"max_moment=q{max_index}:{float(terms['contact_moment'][max_index]):+8.3f} N*m"
                )
        else:
            print("  invalid contact-moment rows (projected |moment| / |force| > 2): none")

        self._print_residual_attribution(unpacked_rows)
        self._print_limit_proximity_diagnostics(unpacked_rows)

        energy_by_dof = [0.0 for _ in range(self.num_dofs)]
        for _row, terms in unpacked_rows:
            for index, value in enumerate(terms["unmodeled"]):
                energy_by_dof[index] += float(value) * float(value)
        total_energy = max(sum(energy_by_dof), 1.0e-12)
        print("  top residual-energy DOFs:")
        side = str(unpacked_rows[0][0][4])
        joint_names = self._joint_names_by_side.get(side, ())
        for index in sorted(range(self.num_dofs), key=lambda dof: energy_by_dof[dof], reverse=True)[:5]:
            joint_label = joint_names[index] if index < len(joint_names) else f"q{index}"
            print(f"    q{index:<2d} {joint_label:<36} {100.0 * energy_by_dof[index] / total_energy:6.2f}%")

        if filtered_inertia_by_sample:
            deltas = []
            raw_deltas = []
            filtered_norms = []
            original_norms = []
            raw_norms = []
            for row, terms in unpacked_rows:
                sample_id = int(row[0])
                if sample_id not in filtered_inertia_by_sample:
                    continue
                filtered = filtered_inertia_by_sample[sample_id]
                original = terms["inertia"]
                recording_interval = terms["inertia_recording_interval"]
                deltas.append(vector_norm(tuple(filtered[i] - original[i] for i in range(self.num_dofs))))
                raw_deltas.append(vector_norm(tuple(recording_interval[i] - original[i] for i in range(self.num_dofs))))
                filtered_norms.append(vector_norm(filtered))
                original_norms.append(vector_norm(original))
                raw_norms.append(vector_norm(recording_interval))
            if deltas:
                print("  inertia estimator check:")
                print(
                    "    M*central_diff(dq) vs recorded tau_inertia: "
                    f"mean_delta={sum(deltas) / len(deltas):8.3f} N*m  "
                    f"max_delta={max(deltas):8.3f} N*m  "
                    f"mean_filtered={sum(filtered_norms) / len(filtered_norms):8.3f} N*m  "
                    f"mean_recorded={sum(original_norms) / len(original_norms):8.3f} N*m"
                )
                print(
                    "    recording-interval full inertia vs recorded tau_inertia: "
                    f"mean_delta={sum(raw_deltas) / len(raw_deltas):8.3f} N*m  "
                    f"max_delta={max(raw_deltas):8.3f} N*m  "
                    f"mean_record_interval={sum(raw_norms) / len(raw_norms):8.3f} N*m"
                )

        rows = [terms for _row, terms in unpacked_rows]
        contact_lag_scores = []
        for lag in (-2, -1, 1, 2):
            percentages = []
            for index, terms in enumerate(rows):
                shifted_index = index + lag
                if shifted_index < 0 or shifted_index >= len(rows):
                    continue
                residual_lag = tuple(
                    terms["inertia"][i]
                    - terms["gravity"][i]
                    - terms["coriolis"][i]
                    - terms["tendon"][i]
                    - terms["actuation"][i]
                    - rows[shifted_index]["contact_validated"][i]
                    - terms["friction"][i]
                    for i in range(self.num_dofs)
                )
                scale = max(
                    vector_norm(terms["inertia"]),
                    vector_norm(terms["gravity"]),
                    vector_norm(terms["coriolis"]),
                    vector_norm(terms["tendon"]),
                    vector_norm(rows[shifted_index]["contact_validated"]),
                    vector_norm(terms["friction"]),
                    1.0,
                )
                percentages.append(100.0 * vector_norm(residual_lag) / scale)
            if percentages:
                contact_lag_scores.append((lag, sum(percentages) / len(percentages)))
        if contact_lag_scores:
            formatted = "  ".join(f"lag {lag:+d}: {score:6.2f}%" for lag, score in contact_lag_scores)
            print(f"  contact timing check mean residual: {formatted}")

    def _print_dynamics_quality_gates(self, unpacked_rows: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
        if not unpacked_rows:
            return

        def vector_norm(values: tuple[float, ...]) -> float:
            return math.sqrt(sum(float(value) * float(value) for value in values))

        def summary(norms: list[float]) -> tuple[float, float, float, float, float]:
            if not norms:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            sorted_norms = sorted(norms)
            return (
                sum(norms) / len(norms),
                sorted_norms[len(sorted_norms) // 2],
                sorted_norms[min(len(sorted_norms) - 1, int(0.90 * (len(sorted_norms) - 1)))],
                sorted_norms[min(len(sorted_norms) - 1, int(0.95 * (len(sorted_norms) - 1)))],
                sorted_norms[-1],
            )

        def best_norm(terms: dict[str, tuple[float, ...]]) -> float:
            return vector_norm(terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"])

        print("  sysid quality gates using hip23+internal diagnostic residual:")
        all_norms = [best_norm(terms) for _row, terms in unpacked_rows]
        mean, median, p90, p95, max_norm = summary(all_norms)
        print(
            f"    all rows       rows={len(all_norms):>5}  "
            f"mean={mean:8.3f} N*m  median={median:8.3f}  p90={p90:8.3f}  p95={p95:8.3f}  max={max_norm:8.3f}"
        )
        for threshold in (100.0, 150.0, 200.0, 300.0, 500.0, 1000.0):
            count = sum(norm <= threshold for norm in all_norms)
            print(
                f"    usable <= {threshold:6.1f} N*m: {count:>5} / "
                f"{len(all_norms):<5} ({100.0 * count / len(all_norms):5.1f}%)"
            )

        issue_checks = (
            (
                "limit penetration >0.02 rad",
                lambda terms: min(float(value) for value in terms["joint_limit_distance_min"]) < -0.02,
            ),
            ("|inertia| >1000 N*m", lambda terms: vector_norm(terms["inertia"]) > 1000.0),
            ("|solver_internal| >1000 N*m", lambda terms: vector_norm(terms["solver_constraint_internal"]) > 1000.0),
            ("|command| >1000 N*m", lambda terms: vector_norm(terms["actuation_command"]) > 1000.0),
            ("|contact_force| >1000 N*m", lambda terms: vector_norm(terms["contact_force"]) > 1000.0),
        )
        print("  sysid issue counters:")
        for label, check in issue_checks:
            count = sum(1 for _row, terms in unpacked_rows if check(terms))
            print(
                f"    {label:<28} rows={count:>5} / {len(unpacked_rows):<5} "
                f"({100.0 * count / len(unpacked_rows):5.1f}%)"
            )

        env_ids = sorted({int(row[3]) for row, _terms in unpacked_rows})
        print("  sysid quality by env:")
        for env_id in env_ids:
            norms = [best_norm(terms) for row, terms in unpacked_rows if int(row[3]) == env_id]
            mean, median, p90, p95, max_norm = summary(norms)
            print(
                f"    env={env_id:>4} rows={len(norms):>4}  "
                f"mean={mean:8.3f}  median={median:8.3f}  p90={p90:8.3f}  p95={p95:8.3f}  max={max_norm:8.3f}"
            )

        print("  worst sysid diagnostic rows:")
        worst_rows = sorted(((best_norm(terms), row, terms) for row, terms in unpacked_rows), reverse=True)[:8]
        joint_names = self._joint_names_by_side.get(str(unpacked_rows[0][0][4]), ())
        for norm, row, terms in worst_rows:
            residual = terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"]
            max_index = max(range(self.num_dofs), key=lambda dof: abs(float(residual[dof])))
            joint_label = joint_names[max_index] if max_index < len(joint_names) else f"q{max_index}"
            print(
                "    "
                f"sample={int(row[0]):>7} env={int(row[3]):>4} step={int(row[1]):>6} "
                f"norm={norm:9.3f} N*m  "
                f"max=q{max_index} {joint_label:<34} {float(residual[max_index]):+9.3f} N*m  "
                f"|inertia|={vector_norm(terms['inertia']):8.3f}  "
                f"|command|={vector_norm(terms['actuation_command']):8.3f}  "
                f"|solver_int|={vector_norm(terms['solver_constraint_internal']):8.3f}  "
                f"|contact|={vector_norm(terms['contact_force']):8.3f}"
            )

    def _print_residual_attribution(self, unpacked_rows: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:  # noqa: C901
        residual_values = [float(value) for _row, terms in unpacked_rows for value in terms["unmodeled"]]
        residual_energy = sum(value * value for value in residual_values)
        if residual_energy <= 1.0e-12:
            return

        def flatten(name: str) -> list[float]:
            return [float(value) for _row, terms in unpacked_rows for value in terms[name]]

        def fit_subtraction(signal: list[float]) -> tuple[float, float, float]:
            signal_energy = sum(value * value for value in signal)
            if signal_energy <= 1.0e-12:
                return 0.0, 0.0, math.sqrt(residual_energy / max(len(residual_values), 1))
            alpha = sum(r * s for r, s in zip(residual_values, signal)) / signal_energy
            after = [r - alpha * s for r, s in zip(residual_values, signal)]
            after_energy = sum(value * value for value in after)
            reduction = 100.0 * (1.0 - after_energy / residual_energy)
            rms_after = math.sqrt(after_energy / max(len(after), 1))
            return alpha, reduction, rms_after

        def residual_norm(row_terms: dict[str, tuple[float, ...]]) -> float:
            return math.sqrt(sum(float(value) * float(value) for value in row_terms["unmodeled"]))

        def per_dof_command_fit() -> tuple[float, float, float, list[dict[str, float]]]:
            before_energy = 0.0
            after_energy = 0.0
            active_alphas = []
            summaries = []
            for dof in range(self.num_dofs):
                residual = [float(terms["unmodeled"][dof]) for _row, terms in unpacked_rows]
                command = [float(terms["actuation_command"][dof]) for _row, terms in unpacked_rows]
                command_energy = sum(value * value for value in command)
                residual_dof_energy = sum(value * value for value in residual)
                command_rms = math.sqrt(command_energy / max(len(command), 1))
                residual_rms = math.sqrt(residual_dof_energy / max(len(residual), 1))
                summaries.append(
                    {
                        "dof": float(dof),
                        "alpha": 0.0,
                        "command_rms": command_rms,
                        "residual_rms": residual_rms,
                        "energy_reduction": 0.0,
                    }
                )
                if command_rms <= 1.0:
                    continue
                alpha = sum(r * c for r, c in zip(residual, command)) / command_energy
                after_dof_energy = sum((r - alpha * c) ** 2 for r, c in zip(residual, command))
                summaries[-1]["alpha"] = alpha
                summaries[-1]["energy_reduction"] = 100.0 * (1.0 - after_dof_energy / max(residual_dof_energy, 1.0e-12))
                active_alphas.append(alpha)
                before_energy += residual_dof_energy
                after_energy += after_dof_energy
            reduction = 100.0 * (1.0 - after_energy / max(before_energy, 1.0e-12))
            mean_abs_alpha = sum(abs(alpha) for alpha in active_alphas) / max(len(active_alphas), 1)
            max_abs_alpha = max((abs(alpha) for alpha in active_alphas), default=0.0)
            return mean_abs_alpha, max_abs_alpha, reduction, summaries

        def print_row_subset_summary(name: str, selected: list[dict[str, tuple[float, ...]]]) -> None:
            if not selected:
                print(f"    {name:<14} rows=0")
                return
            residual_norms = [residual_norm(terms) for terms in selected]
            contact_norms = [
                math.sqrt(sum(float(value) * float(value) for value in terms["contact_validated"]))
                for terms in selected
            ]
            command_norms = [
                math.sqrt(sum(float(value) * float(value) for value in terms["actuation_command"]))
                for terms in selected
            ]
            residual_norms_sorted = sorted(residual_norms)
            p95 = residual_norms_sorted[
                min(len(residual_norms_sorted) - 1, int(0.95 * (len(residual_norms_sorted) - 1)))
            ]
            print(
                f"    {name:<14} rows={len(selected):>4}  "
                f"mean|r|={sum(residual_norms) / len(residual_norms):8.3f} N*m  "
                f"p95|r|={p95:8.3f} N*m  "
                f"mean|contact|={sum(contact_norms) / len(contact_norms):8.3f} N*m  "
                f"mean|command|={sum(command_norms) / len(command_norms):8.3f} N*m"
            )

        def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
            size = len(vector)
            augmented = [list(row) + [float(vector[index])] for index, row in enumerate(matrix)]
            for pivot_index in range(size):
                pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
                pivot = augmented[pivot_row][pivot_index]
                if abs(pivot) <= 1.0e-10:
                    return None
                if pivot_row != pivot_index:
                    augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
                scale = augmented[pivot_index][pivot_index]
                for col in range(pivot_index, size + 1):
                    augmented[pivot_index][col] /= scale
                for row in range(size):
                    if row == pivot_index:
                        continue
                    factor = augmented[row][pivot_index]
                    if factor == 0.0:
                        continue
                    for col in range(pivot_index, size + 1):
                        augmented[row][col] -= factor * augmented[pivot_index][col]
            return [augmented[row][size] for row in range(size)]

        def multivariate_fit(signal_names: tuple[str, ...]) -> tuple[list[float] | None, float, float]:
            signals = [flatten(name) for name in signal_names]
            matrix = [
                [sum(left * right for left, right in zip(signals[row], signals[col])) for col in range(len(signals))]
                for row in range(len(signals))
            ]
            vector = [sum(r * value for r, value in zip(residual_values, signal)) for signal in signals]
            coefficients = solve_linear_system(matrix, vector)
            if coefficients is None:
                return None, 0.0, math.sqrt(residual_energy / max(len(residual_values), 1))
            after = [
                residual - sum(coefficients[col] * signals[col][index] for col in range(len(signal_names)))
                for index, residual in enumerate(residual_values)
            ]
            after_energy = sum(value * value for value in after)
            reduction = 100.0 * (1.0 - after_energy / residual_energy)
            rms_after = math.sqrt(after_energy / max(len(after), 1))
            return coefficients, reduction, rms_after

        print("  residual attribution fits (diagnostic only, not used in tau_unmodeled):")
        for name in (
            "actuation_command",
            "contact_force",
            "contact_moment",
            "contact_validated",
            "contact_digit",
            "contact_digit_force",
            "contact_digit_moment",
            "contact_connector",
            "contact_connector_force",
            "contact_connector_moment",
            "contact_base",
            "contact_base_force",
            "contact_base_moment",
            "tendon",
            "friction_dynamic",
            "friction_viscous",
            "actuation_estimated_hip",
            "actuation_estimated_hip_lateral_flexion",
            "actuation_estimated_passive",
            "drive_stiffness",
            "drive_damping",
            "drive_pd_clipped",
            "solver_constraint_passive",
            "solver_constraint_limit",
            "solver_constraint_internal",
            "armature_inertia",
            "inertia_recording_interval",
            "inertia_raw",
        ):
            alpha, reduction, rms_after = fit_subtraction(flatten(name))
            print(
                f"    subtract {name:<26} alpha={alpha:+7.3f}  "
                f"energy_reduction={reduction:7.2f}%  rms_after={rms_after:8.3f} N*m/dof"
            )
        mean_abs_alpha, max_abs_alpha, reduction, dof_command_summaries = per_dof_command_fit()
        print(
            "    subtract actuation_command per active-command DOF "
            f"mean|alpha|={mean_abs_alpha:6.3f}  max|alpha|={max_abs_alpha:6.3f}  "
            f"energy_reduction={reduction:7.2f}%"
        )
        print("  multivariate residual fits (diagnostic only):")
        for names in (
            ("actuation_command", "contact_validated"),
            ("actuation_command", "contact_force", "contact_moment"),
            ("actuation_command", "contact_validated", "inertia_recording_interval"),
            ("actuation_estimated_hip", "contact_validated"),
            ("actuation_estimated_hip", "contact_force", "contact_moment"),
            ("actuation_estimated_hip", "contact_force", "solver_constraint_passive"),
            ("actuation_estimated_hip", "contact_validated", "solver_constraint_passive"),
            ("actuation_estimated_hip", "contact_force", "contact_moment", "solver_constraint_passive"),
            ("actuation_estimated_hip_lateral_flexion", "contact_force", "solver_constraint_internal"),
            (
                "actuation_estimated_hip_lateral_flexion",
                "contact_digit_force",
                "contact_connector_force",
                "contact_base_force",
                "solver_constraint_internal",
            ),
            (
                "actuation_estimated_hip_lateral_flexion",
                "contact_force",
                "contact_moment",
                "solver_constraint_internal",
            ),
            ("drive_pd_clipped", "contact_force", "solver_constraint_internal"),
            ("drive_stiffness", "drive_damping", "contact_force", "solver_constraint_internal"),
            ("actuation_estimated_hip_lateral_flexion", "contact_force", "solver_constraint_internal", "friction"),
            (
                "actuation_estimated_hip_lateral_flexion",
                "contact_force",
                "solver_constraint_internal",
                "armature_inertia",
            ),
        ):
            coefficients, reduction, rms_after = multivariate_fit(names)
            if coefficients is None:
                print(f"    {' + '.join(names)}: singular fit")
                continue
            formatted_coefficients = ", ".join(
                f"{name}={coefficient:+.3f}" for name, coefficient in zip(names, coefficients)
            )
            print(
                f"    subtract {formatted_coefficients}  "
                f"energy_reduction={reduction:7.2f}%  rms_after={rms_after:8.3f} N*m/dof"
            )
        side = str(unpacked_rows[0][0][4])
        joint_names = self._joint_names_by_side.get(side, ())
        print("  per-DOF command fit, active command channels only (|command| RMS > 1 N*m):")
        active_summaries = [item for item in dof_command_summaries if item["command_rms"] > 1.0]
        active_summaries.sort(key=lambda item: item["energy_reduction"], reverse=True)
        for item in active_summaries[:8]:
            dof = int(item["dof"])
            joint_label = joint_names[dof] if dof < len(joint_names) else f"q{dof}"
            print(
                f"    q{dof:<2d} {joint_label:<36} "
                f"cmd_rms={item['command_rms']:8.3f} N*m  "
                f"res_rms={item['residual_rms']:8.3f} N*m  "
                f"alpha={item['alpha']:+7.3f}  "
                f"energy_reduction={item['energy_reduction']:7.2f}%"
            )

        print("  per-DOF contact fit after hip23+internal diagnostic:")
        contact_summaries = []
        contact_fit_terms = ("contact_force", "contact_digit_force", "contact_connector_force", "contact_base_force")
        for dof in range(self.num_dofs):
            final_residual = []
            for _row, terms in unpacked_rows:
                best_residual = float(
                    terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"][dof]
                )
                final_residual.append(best_residual)
            final_rms = math.sqrt(sum(value * value for value in final_residual) / max(len(final_residual), 1))
            best_fit = ("none", 0.0, 0.0, final_rms, 0.0)
            for contact_term in contact_fit_terms:
                residual_before_contact = []
                contact_force = []
                for _row, terms in unpacked_rows:
                    best_residual = float(
                        terms["unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal"][dof]
                    )
                    contact_value = float(terms[contact_term][dof])
                    residual_before_contact.append(best_residual + contact_value)
                    contact_force.append(contact_value)
                contact_energy = sum(value * value for value in contact_force)
                before_energy = sum(value * value for value in residual_before_contact)
                contact_rms = math.sqrt(contact_energy / max(len(contact_force), 1))
                if contact_energy <= 1.0e-12 or before_energy <= 1.0e-12:
                    alpha = 0.0
                    reduction = 0.0
                    fitted_rms = math.sqrt(before_energy / max(len(residual_before_contact), 1))
                else:
                    alpha = sum(r * c for r, c in zip(residual_before_contact, contact_force)) / contact_energy
                    after = [r - alpha * c for r, c in zip(residual_before_contact, contact_force)]
                    after_energy = sum(value * value for value in after)
                    reduction = 100.0 * (1.0 - after_energy / before_energy)
                    fitted_rms = math.sqrt(after_energy / max(len(after), 1))
                if reduction > best_fit[2]:
                    best_fit = (contact_term, alpha, reduction, fitted_rms, contact_rms)
            contact_summaries.append((final_rms, dof, *best_fit))
        for final_rms, dof, contact_term, alpha, reduction, fitted_rms, contact_rms in sorted(
            contact_summaries, reverse=True
        )[:8]:
            joint_label = joint_names[dof] if dof < len(joint_names) else f"q{dof}"
            print(
                f"    q{dof:<2d} {joint_label:<36} "
                f"current_rms={final_rms:8.3f} N*m  "
                f"best_contact={contact_term:<23} rms={contact_rms:8.3f} N*m  "
                f"best_alpha={alpha:+7.3f}  "
                f"fit_rms={fitted_rms:8.3f} N*m  "
                f"energy_reduction={reduction:7.2f}%"
            )

        contact_rows = []
        no_contact_rows = []
        for _row, terms in unpacked_rows:
            contact_norm = math.sqrt(sum(float(value) * float(value) for value in terms["contact_validated"]))
            if contact_norm > 1.0:
                contact_rows.append(terms)
            else:
                no_contact_rows.append(terms)
        print("  residual split by contact activity:")
        print_row_subset_summary("contact", contact_rows)
        print_row_subset_summary("no contact", no_contact_rows)

    def _print_limit_proximity_diagnostics(self, unpacked_rows: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
        if not unpacked_rows:
            return

        def vector_norm(values: tuple[float, ...]) -> float:
            return math.sqrt(sum(float(value) * float(value) for value in values))

        side = str(unpacked_rows[0][0][4])
        joint_names = self._joint_names_by_side.get(side, ())
        print("  per-DOF joint-limit residual split (limit distance <= 0.05 rad):")
        limit_summaries = []
        for dof in range(self.num_dofs):
            near = [terms for _row, terms in unpacked_rows if float(terms["joint_limit_distance_min"][dof]) <= 0.05]
            far = [terms for _row, terms in unpacked_rows if float(terms["joint_limit_distance_min"][dof]) > 0.05]
            if not near:
                continue
            near_residual = math.sqrt(sum(float(terms["unmodeled"][dof]) ** 2 for terms in near) / len(near))
            far_residual = (
                math.sqrt(sum(float(terms["unmodeled"][dof]) ** 2 for terms in far) / len(far)) if far else 0.0
            )
            near_solver = math.sqrt(
                sum(float(terms["solver_constraint_limit"][dof]) ** 2 for terms in near) / len(near)
            )
            min_distance = min(float(terms["joint_limit_distance_min"][dof]) for _row, terms in unpacked_rows)
            limit_summaries.append((near_residual, dof, len(near), len(far), far_residual, near_solver, min_distance))
        if limit_summaries:
            for near_residual, dof, near_count, far_count, far_residual, near_solver, min_distance in sorted(
                limit_summaries, reverse=True
            )[:8]:
                joint_label = joint_names[dof] if dof < len(joint_names) else f"q{dof}"
                print(
                    f"    q{dof:<2d} {joint_label:<36} "
                    f"near={near_count:>4} far={far_count:>4}  "
                    f"near_res_rms={near_residual:8.3f} N*m  "
                    f"far_res_rms={far_residual:8.3f} N*m  "
                    f"limit_solver_rms={near_solver:8.3f} N*m  "
                    f"min_dist={min_distance:8.4f} rad"
                )
        else:
            print("    none")

        print("  closest joint-limit distances by DOF:")
        summaries = []
        for dof in range(self.num_dofs):
            min_distance = min(float(terms["joint_limit_distance_min"][dof]) for _row, terms in unpacked_rows)
            residual_rms = math.sqrt(
                sum(float(terms["unmodeled"][dof]) ** 2 for _row, terms in unpacked_rows) / len(unpacked_rows)
            )
            solver_rms = math.sqrt(
                sum(float(terms["solver_constraint_limit"][dof]) ** 2 for _row, terms in unpacked_rows)
                / len(unpacked_rows)
            )
            summaries.append((min_distance, residual_rms, solver_rms, dof))
        for min_distance, residual_rms, solver_rms, dof in sorted(summaries, key=lambda item: item[0])[:6]:
            joint_label = joint_names[dof] if dof < len(joint_names) else f"q{dof}"
            print(
                f"    q{dof:<2d} {joint_label:<36} "
                f"min_dist={min_distance:8.4f} rad  "
                f"res_rms={residual_rms:8.3f} N*m  "
                f"limit_solver_rms={solver_rms:8.3f} N*m"
            )

    def _joint_metadata(self) -> list[dict[str, Any]]:
        rows = []
        for side, names in self._joint_names_by_side.items():
            for q_index, (joint_name, joint_index) in enumerate(zip(names, self._joint_indices_by_side[side])):
                rows.append(
                    {
                        "side": side,
                        "q_index": q_index,
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


def _debug_dynamics_data_columns(num_dofs: int) -> list[str]:
    return (
        ["sample_id", "step_index", "time", "env_id", "side"]
        + [f"tau_inertia{i}" for i in range(num_dofs)]
        + [f"tau_inertia_recording_interval{i}" for i in range(num_dofs)]
        + [f"tau_inertia_raw{i}" for i in range(num_dofs)]
        + [f"tau_inertia_joint_only{i}" for i in range(num_dofs)]
        + [f"tau_inertia_joint_all{i}" for i in range(num_dofs)]
        + [f"tau_inertia_leg_self{i}" for i in range(num_dofs)]
        + [f"tau_inertia_other_joints{i}" for i in range(num_dofs)]
        + [f"tau_inertia_root_coupling{i}" for i in range(num_dofs)]
        + [f"tau_inertia_root_coupling_raw{i}" for i in range(num_dofs)]
        + [f"tau_inertia_root_coupling_alt{i}" for i in range(num_dofs)]
        + [f"tau_inertia_root_coupled_alt{i}" for i in range(num_dofs)]
        + [f"tau_inertia_full_raw{i}" for i in range(num_dofs)]
        + [f"tau_coriolis{i}" for i in range(num_dofs)]
        + [f"tau_gravity{i}" for i in range(num_dofs)]
        + [f"tau_friction_dynamic{i}" for i in range(num_dofs)]
        + [f"tau_friction_viscous{i}" for i in range(num_dofs)]
        + [f"tau_friction{i}" for i in range(num_dofs)]
        + [f"tau_solver_joint{i}" for i in range(num_dofs)]
        + [f"tau_actuation{i}" for i in range(num_dofs)]
        + [f"tau_actuation_command{i}" for i in range(num_dofs)]
        + [f"tau_actuation_estimated{i}" for i in range(num_dofs)]
        + [f"tau_actuation_estimated_hip{i}" for i in range(num_dofs)]
        + [f"tau_actuation_estimated_hip_lateral_flexion{i}" for i in range(num_dofs)]
        + [f"tau_actuation_estimated_passive{i}" for i in range(num_dofs)]
        + [f"tau_physx_actuation{i}" for i in range(num_dofs)]
        + [f"tau_solver_constraint_passive{i}" for i in range(num_dofs)]
        + [f"tau_solver_constraint_limit{i}" for i in range(num_dofs)]
        + [f"tau_solver_constraint_internal{i}" for i in range(num_dofs)]
        + [f"tau_joint_drive_pos_target{i}" for i in range(num_dofs)]
        + [f"tau_joint_drive_vel_target{i}" for i in range(num_dofs)]
        + [f"tau_joint_drive_effort_target{i}" for i in range(num_dofs)]
        + [f"tau_joint_drive_stiffness{i}" for i in range(num_dofs)]
        + [f"tau_joint_drive_damping{i}" for i in range(num_dofs)]
        + [f"tau_joint_effort_limit{i}" for i in range(num_dofs)]
        + [f"tau_joint_velocity_limit{i}" for i in range(num_dofs)]
        + [f"tau_joint_limit_lower{i}" for i in range(num_dofs)]
        + [f"tau_joint_limit_upper{i}" for i in range(num_dofs)]
        + [f"tau_joint_limit_distance_lower{i}" for i in range(num_dofs)]
        + [f"tau_joint_limit_distance_upper{i}" for i in range(num_dofs)]
        + [f"tau_joint_limit_distance_min{i}" for i in range(num_dofs)]
        + [f"tau_drive_stiffness{i}" for i in range(num_dofs)]
        + [f"tau_drive_damping{i}" for i in range(num_dofs)]
        + [f"tau_drive_effort_target{i}" for i in range(num_dofs)]
        + [f"tau_drive_pd{i}" for i in range(num_dofs)]
        + [f"tau_drive_pd_clipped{i}" for i in range(num_dofs)]
        + [f"tau_armature_inertia{i}" for i in range(num_dofs)]
        + [f"tau_contact{i}" for i in range(num_dofs)]
        + [f"tau_contact_force{i}" for i in range(num_dofs)]
        + [f"tau_contact_moment{i}" for i in range(num_dofs)]
        + [f"tau_contact_validated{i}" for i in range(num_dofs)]
        + [f"tau_contact_digit{i}" for i in range(num_dofs)]
        + [f"tau_contact_digit_force{i}" for i in range(num_dofs)]
        + [f"tau_contact_digit_moment{i}" for i in range(num_dofs)]
        + [f"tau_contact_connector{i}" for i in range(num_dofs)]
        + [f"tau_contact_connector_force{i}" for i in range(num_dofs)]
        + [f"tau_contact_connector_moment{i}" for i in range(num_dofs)]
        + [f"tau_contact_base{i}" for i in range(num_dofs)]
        + [f"tau_contact_base_force{i}" for i in range(num_dofs)]
        + [f"tau_contact_base_moment{i}" for i in range(num_dofs)]
        + [f"tau_tendon{i}" for i in range(num_dofs)]
        + [f"tau_tendon_model{i}" for i in range(num_dofs)]
        + [f"tau_tendon_projection_delta{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_quasistatic{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_full_dynamics{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_recording_interval{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_estimated_actuation{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_estimated_hip_actuation{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_estimated_hip_force_contact{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_estimated_hip_force_contact_solver{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_estimated_hip_lateral_flexion_force_contact_solver_internal{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_full_contact{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_contact_force_only{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled_contact_validated{i}" for i in range(num_dofs)]
        + [f"tau_unmodeled{i}" for i in range(num_dofs)]
        + [f"tau_inverse_residual{i}" for i in range(num_dofs)]
        + [f"tau_solver_residual{i}" for i in range(num_dofs)]
        + list(DEBUG_DYNAMICS_SCALAR_NAMES)
        + [f"mass_matrix{row}_{col}" for row in range(num_dofs) for col in range(num_dofs)]
    )


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


def _jsonable_config(cfg: DataRecordingConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["output_dir"] = str(data["output_dir"])
    return data
