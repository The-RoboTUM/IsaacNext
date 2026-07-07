# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identix-compatible data recording helpers for Forrest tendon simulations."""

from __future__ import annotations

import json
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
        self.metadata_path = self.output_dir / cfg.metadata_filename
        self.viz_vars_path = self.output_dir / cfg.viz_vars_filename

        self._db: sqlite3.Connection | None = None
        self._tendon_db: sqlite3.Connection | None = None
        self._dynamics_db: sqlite3.Connection | None = None
        self._sim_buffer: list[tuple[float, ...]] = []
        self._tendon_buffer: list[tuple[int, float, str, str]] = []
        self._dynamics_buffer: list[tuple[Any, ...]] = []
        self._joint_indices_by_side: dict[str, list[int]] = {}
        self._joint_names_by_side: dict[str, tuple[str, ...]] = {}
        self._body_indices_by_side: dict[str, list[int]] = {}
        self._body_names_by_side: dict[str, tuple[str, ...]] = {}
        self._joint_dynamics_properties_rows: list[dict[str, Any]] = []
        self._sim_columns: list[str] = []
        self._dynamics_columns: list[str] = []
        self._spatial_columns: list[str] = []
        self._row_count = 0
        self._tendon_row_count = 0
        self._dynamics_row_count = 0
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
            existing = [
                path
                for path in (
                    self.sqlite_path,
                    self.tendon_sqlite_path,
                    self.dynamics_sqlite_path,
                    self.metadata_path,
                    self.viz_vars_path,
                )
                if path.exists()
            ]
            if existing:
                raise FileExistsError(f"Recording output already exists: {existing}")
        for path in (
            self.sqlite_path,
            self.tendon_sqlite_path,
            self.dynamics_sqlite_path,
            self.metadata_path,
            self.viz_vars_path,
        ):
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

        if not self.cfg.record_dynamics:
            return
        if not self._initialized or self._dynamics_db is None:
            raise RuntimeError("Call initialize(...) before record_dynamics_step(...).")
        if self._closed:
            raise RuntimeError("Cannot record after close().")
        if sim_time < self.cfg.startup_skip_seconds:
            return
        if step_index % self.cfg.sampling_stride != 0:
            return

        required = ("inertia", "coriolis", "gravity", "friction")
        missing = [name for name in required if name not in dynamics_terms]
        if missing:
            raise ValueError(f"Missing dynamics term tensors: {missing}")

        tau_total = (
            dynamics_terms["inertia"]
            + dynamics_terms["coriolis"]
            + dynamics_terms["gravity"]
            + dynamics_terms["friction"]
        )
        tau_residual = tau_total - tau_input

        for env_id in self._selected_env_ids:
            for side in self._selected_sides():
                joint_indices = self._joint_indices_by_side[side]
                row_values: list[Any] = [
                    int(self._dynamics_row_count),
                    int(step_index),
                    float(sim_time),
                    int(env_id),
                    side,
                ]
                for term_name in ("inertia", "coriolis", "gravity", "friction"):
                    term_values = dynamics_terms[term_name][env_id, joint_indices].detach().cpu().tolist()
                    row_values.extend(float(value) for value in term_values)
                row_values.extend(float(value) for value in tau_total[env_id, joint_indices].detach().cpu().tolist())
                row_values.extend(float(value) for value in tau_residual[env_id, joint_indices].detach().cpu().tolist())
                self._dynamics_buffer.append(tuple(row_values))
                self._dynamics_row_count += 1

        if len(self._dynamics_buffer) >= self.cfg.batch_size:
            self.flush()

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
        ddq_all = robot.data.joint_acc
        tau_all = self._tau_tensor(robot, tau_override=tau_override)
        context = dict(extra_context or {})

        for env_id in self._selected_env_ids:
            for side in self._selected_sides():
                joint_indices = self._joint_indices_by_side[side]
                q = q_all[env_id, joint_indices].detach().cpu().tolist()
                dq = dq_all[env_id, joint_indices].detach().cpu().tolist()
                ddq = ddq_all[env_id, joint_indices].detach().cpu().tolist()
                tau = tau_all[env_id, joint_indices].detach().cpu().tolist()

                self._sim_buffer.append(tuple(float(value) for value in [*q, *dq, *ddq, *tau]))
                self._row_count += 1

        if context:
            self._context_metadata.setdefault("per_step_context_seen", sorted(context))
        if len(self._sim_buffer) >= self.cfg.batch_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered rows to SQLite."""

        if self._db is None:
            return
        if self._sim_buffer:
            placeholders = ", ".join("?" for _ in self._sim_columns)
            columns = ", ".join(_quote_identifier(name) for name in self._sim_columns)
            self._db.executemany(
                f"INSERT INTO {_quote_identifier(self.cfg.sim_table_name)} ({columns}) VALUES ({placeholders})",
                self._sim_buffer,
            )
            self._sim_buffer.clear()
        if self._tendon_db is not None and self._tendon_buffer:
            self._tendon_db.executemany(
                "INSERT INTO tendon_frames (step_index, time, side, frame_json) VALUES (?, ?, ?, ?)",
                self._tendon_buffer,
            )
            self._tendon_buffer.clear()
        if self._dynamics_db is not None and self._dynamics_buffer:
            placeholders = ", ".join("?" for _ in self._dynamics_columns)
            columns = ", ".join(_quote_identifier(name) for name in self._dynamics_columns)
            self._dynamics_db.executemany(
                f"INSERT INTO dynamics_data ({columns}) VALUES ({placeholders})",
                self._dynamics_buffer,
            )
            self._dynamics_buffer.clear()
        self._db.commit()
        if self._tendon_db is not None:
            self._tendon_db.commit()
        if self._dynamics_db is not None:
            self._dynamics_db.commit()

    def close(self) -> None:
        """Flush rows, write metadata, and close the SQLite connection."""

        if self._closed:
            return
        self.flush()
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
            self._dynamics_columns = _dynamics_data_columns(self.num_dofs)
            self._dynamics_db = sqlite3.connect(self.dynamics_sqlite_path)
            columns_sql = ", ".join(_dynamics_column_sql(name) for name in self._dynamics_columns)
            self._dynamics_db.execute(f"CREATE TABLE dynamics_data ({columns_sql})")
            self._dynamics_db.execute("CREATE INDEX dynamics_data_step_idx ON dynamics_data (step_index, side)")
            self._dynamics_db.commit()

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
            "viz_vars_path": str(self.viz_vars_path),
            "sim_table_name": self.cfg.sim_table_name,
            "dynamics_table_name": "dynamics_data" if self.cfg.record_dynamics else None,
            "num_dofs": self.num_dofs,
            "row_count": self._row_count,
            "tendon_row_count": self._tendon_row_count,
            "dynamics_row_count": self._dynamics_row_count,
            "sim_columns": self._sim_columns,
            "dynamics_columns": self._dynamics_columns,
            "sim_dt": self._sim_dt,
            "selected_env_ids": list(self._selected_env_ids),
            "sample_order": "for each recorded step: selected_env_ids in order, then selected_sides in order",
            "tau_source": self.cfg.tau_source,
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
            "metadata": self.cfg.metadata_filename,
            "sim_table_name": self.cfg.sim_table_name,
            "tendon_table_name": "tendon_frames",
            "dynamics_table_name": "dynamics_data",
            "tendon_frame_format": "jsonl-compatible analytic tendon debug frame",
            "num_dofs": self.num_dofs,
            "sim_dt": self._sim_dt,
            "selected_sides": list(self._selected_sides()),
            "selected_env_ids": list(self._selected_env_ids),
            "sample_order": "for each recorded step: selected_env_ids in order, then selected_sides in order",
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
                "from contact sensor forces projected with J^T f; tendon forces are intentionally excluded"
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
            "sample_id": "zero-based row index aligned one-to-one with sim_data rowid - 1",
            "tau_inertia": "selected rows of PhysX generalized mass matrix multiplied by IsaacLab joint_acc",
            "tau_coriolis": "PhysX Coriolis and centrifugal compensation forces for the current articulation state",
            "tau_gravity": "PhysX generalized gravity compensation forces for the current articulation pose",
            "tau_friction": (
                "model estimate from configured joint dynamic and viscous friction coefficients; static friction is "
                "stored in metadata because its active solver value is not exposed as a separated generalized force"
            ),
            "tau_model": "tau_inertia + tau_coriolis + tau_gravity + tau_friction",
            "tau_tendon_residual": (
                "tau_model - sim_data tau; intended as the residual target for tendon potential learning"
            ),
        }

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


def _dynamics_data_columns(num_dofs: int) -> list[str]:
    return (
        ["sample_id", "step_index", "time", "env_id", "side"]
        + [f"tau_inertia{i}" for i in range(num_dofs)]
        + [f"tau_coriolis{i}" for i in range(num_dofs)]
        + [f"tau_gravity{i}" for i in range(num_dofs)]
        + [f"tau_friction{i}" for i in range(num_dofs)]
        + [f"tau_model{i}" for i in range(num_dofs)]
        + [f"tau_tendon_residual{i}" for i in range(num_dofs)]
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
