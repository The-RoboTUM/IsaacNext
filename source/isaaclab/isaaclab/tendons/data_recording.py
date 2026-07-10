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

from isaaclab.tendons.models.analytic.constants import (
    actuated_joint_names,
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
    "coriolis",
    "gravity",
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
    "friction",
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
    tau_source: str = "motor_torque"
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
                for name in (*DEBUG_DYNAMICS_TERM_NAMES, *DEBUG_DYNAMICS_MATRIX_TERM_NAMES)
                if name not in dynamics_terms
            ]
            if debug_missing:
                raise ValueError(f"Missing debug dynamics term tensors: {debug_missing}")

        excluded_env_ids = set(skip_env_ids or ())
        for env_id in self._selected_env_ids:
            if int(env_id) in excluded_env_ids:
                continue
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
            values = dynamics_terms["coriolis"][env_id, joint_indices]
        elif term_name == "gravity":
            values = dynamics_terms["gravity"][env_id, joint_indices]
        elif term_name == "tendon":
            values = dynamics_terms["tendon"][env_id, joint_indices]
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

        residual_norm = selected_norm("residual")
        contact_norm = selected_norm("contact_force")
        inertia_norm = selected_norm("inertia")
        actuation_norm = selected_norm("actuation_command")
        return {
            "quality_residual_norm": residual_norm,
            "quality_contact_norm": contact_norm,
            "quality_inertia_norm": inertia_norm,
            "quality_actuation_norm": actuation_norm,
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
        ddq_all = robot.data.joint_acc if ddq_override is None else ddq_override
        tau_all = self._tau_tensor(robot, tau_override=tau_override)
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
        residual = dynamics_terms["residual"][env_id, joint_indices].detach()
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
            "available_tau_sources": [
                "motor_torque",
                "controller_plus_ground",
                "applied_torque",
                "computed_torque",
                "zero",
            ],
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
            "tau_inertia": "selected leg rows of the full floating-base generalized inertia product M(q)qdd",
            "tau_coriolis": "Coriolis/centrifugal term on the left-hand side of the training equation",
            "tau_gravity": "gravity term on the left-hand side of the training equation",
            "tau_tendon": "analytic tendon term on the left-hand side of the training equation",
            "tau_actuation": (
                "external actuation generalized force used by the dynamics force balance; contains motor torque "
                "except pantograph implicit effort and knee-flexor effort, which are kept as debug diagnostics"
            ),
            "tau_contact": "external validated contact generalized force projected from measured contact wrench",
            "tau_friction": "external configured joint-friction model term",
            "tau_external": "tau_actuation + tau_contact + tau_friction",
        }

    def _report_dynamics_residual(self) -> None:
        if not self.cfg.record_debug_dynamics or self._debug_db is None or self._debug_row_count == 0:
            return

        residual_cols = [f"tau_residual{i}" for i in range(self.num_dofs)]
        contact_cols = [f"tau_contact_validated{i}" for i in range(self.num_dofs)]
        actuation_cols = [f"tau_actuation_command{i}" for i in range(self.num_dofs)]
        candidate_terms = (
            "residual_with_pantograph_actuation",
            "residual_with_knee_flexor_actuation",
            "residual_with_pantograph_and_knee_flexor_actuation",
            "residual_no_pantograph_actuation",
            "residual_no_knee_flexor_actuation",
            "residual_no_pantograph_no_knee_flexor_actuation",
            "residual_no_pantograph_no_knee_flexor_plus_solver",
        )
        available_candidate_terms = [
            term
            for term in candidate_terms
            if all(f"tau_{term}{i}" in self._debug_columns for i in range(self.num_dofs))
        ]
        candidate_cols = [f"tau_{term}{i}" for term in available_candidate_terms for i in range(self.num_dofs)]
        selected_cols = [
            "sample_id",
            "step_index",
            "env_id",
            "side",
            *residual_cols,
            *contact_cols,
            *actuation_cols,
            *candidate_cols,
        ]
        col_index = {name: index for index, name in enumerate(selected_cols)}
        quoted_cols = ", ".join(_quote_identifier(name) for name in selected_cols)
        rows = self._debug_db.execute(f"SELECT {quoted_cols} FROM debug_data ORDER BY sample_id").fetchall()
        if not rows:
            return

        def vector_norm(values: tuple[float, ...]) -> float:
            return math.sqrt(sum(value * value for value in values))

        residual_norms = []
        contact_norms = []
        actuation_norms = []
        candidate_norms = {term: [] for term in available_candidate_terms}
        worst = None
        for row in rows:
            residual = tuple(float(row[col_index[f"tau_residual{i}"]]) for i in range(self.num_dofs))
            contact = tuple(float(row[col_index[f"tau_contact_validated{i}"]]) for i in range(self.num_dofs))
            actuation = tuple(float(row[col_index[f"tau_actuation_command{i}"]]) for i in range(self.num_dofs))
            residual_norm = vector_norm(residual)
            residual_norms.append(residual_norm)
            contact_norms.append(vector_norm(contact))
            actuation_norms.append(vector_norm(actuation))
            for term in available_candidate_terms:
                candidate = tuple(float(row[col_index[f"tau_{term}{i}"]]) for i in range(self.num_dofs))
                candidate_norms[term].append(vector_norm(candidate))
            if worst is None or residual_norm > worst[0]:
                worst = (residual_norm, row, residual)

        def mean(values: list[float]) -> float:
            return sum(values) / max(len(values), 1)

        residual_sorted = sorted(residual_norms)
        p95_index = min(len(residual_sorted) - 1, int(0.95 * (len(residual_sorted) - 1)))
        self._context_metadata["dynamics_residual_summary"] = {
            "rows": len(rows),
            "equation": "residual = conservative - non_conservative",
            "mean_residual_norm_nm": mean(residual_norms),
            "p95_residual_norm_nm": residual_sorted[p95_index],
            "max_residual_norm_nm": max(residual_norms),
            "mean_contact_norm_nm": mean(contact_norms),
            "mean_actuation_norm_nm": mean(actuation_norms),
            "candidate_residual_norms_nm": {
                term: {
                    "mean": mean(norms),
                    "p95": sorted(norms)[min(len(norms) - 1, int(0.95 * (len(norms) - 1)))],
                    "max": max(norms),
                }
                for term, norms in candidate_norms.items()
                if norms
            },
        }

        print("\n[ForrestDynamics] Minimal force-balance check")
        print("  residual = conservative - non_conservative")
        print("  conservative = inertia + gravity + coriolis + tendon")
        print("  non_conservative = actuation + contact_validated + friction")
        print("  dynamics actuation excludes pantograph and knee-flexor diagnostics; sim_data tau is motor-only")
        print(f"  rows: {len(rows):,}")
        print(
            "  residual norm N*m: "
            f"mean={mean(residual_norms):.3f}, p95={residual_sorted[p95_index]:.3f}, max={max(residual_norms):.3f}"
        )
        if candidate_norms:
            print("  candidate residual norms N*m:")
            for term, norms in candidate_norms.items():
                if not norms:
                    continue
                sorted_norms = sorted(norms)
                candidate_p95_index = min(len(sorted_norms) - 1, int(0.95 * (len(sorted_norms) - 1)))
                print(
                    f"    {term}: "
                    f"mean={mean(norms):.3f}, p95={sorted_norms[candidate_p95_index]:.3f}, max={max(norms):.3f}"
                )
        if worst is not None:
            _, row, residual = worst
            print(f"  worst sample_id={int(row[0])} step={int(row[1])} env={int(row[2])} side={row[3]}")
            print("  worst residual by DOF:")
            joint_names = self._joint_names_by_side.get(str(row[3]), ())
            for dof, value in enumerate(residual):
                joint_label = joint_names[dof] if dof < len(joint_names) else f"q{dof}"
                print(f"    q{dof:<2d} {joint_label:<36} {value:+10.3f} N*m")
        return

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
        + [f"tau_{name}{i}" for name in DEBUG_DYNAMICS_TERM_NAMES for i in range(num_dofs)]
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


def _jsonable_config(cfg: DataRecordingConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["output_dir"] = str(data["output_dir"])
    return data
