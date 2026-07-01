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
TENDON_CHAIN_LINKS: dict[str, tuple[str, ...]] = {
    "left": tuple(link_names_left),
    "right": tuple(link_names_right),
}
OMITTED_FIRST_PASS_JOINTS: dict[str, tuple[str, ...]] = {
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
}


@dataclass
class DataRecordingConfig:
    """Configuration for one Identix-style recording run."""

    output_dir: str | Path
    sqlite_filename: str = "forrest_tendon_chain_sim_data.db"
    metadata_filename: str = "metadata.json"
    sim_table_name: str = "sim_data"
    context_table_name: str = "sample_context"
    spatial_table_name: str = "spatial_data"
    joint_set: str = "tendon_chain_5"
    side_policy: str = "left_only"
    selected_joint_names: tuple[str, ...] | None = None
    body_set: str = "tendon_chain_links"
    selected_body_names: tuple[str, ...] | None = None
    record_spatial_state: bool = True
    sampling_stride: int = 1
    startup_skip_seconds: float = 0.0
    constraint_mode: str = "static"
    controller: str | None = "sin"
    tau_source: str = "applied_torque"
    overwrite: bool = False
    batch_size: int = 512
    parameter_file: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class DataRecording:
    """Write Forrest simulation samples in Identix ``sim_data`` format.

    ``sim_data`` intentionally contains only positional Identix columns:
    ``q0..qN``, ``dq0..dqN``, ``ddq0..ddqN``, ``tau0..tauN``. Time, env, side,
    joint names, and 3D spatial state are stored separately so Identix loaders
    can read the main table without schema changes.
    """

    def __init__(self, cfg: DataRecordingConfig):
        if cfg.sampling_stride < 1:
            raise ValueError("sampling_stride must be >= 1.")
        if cfg.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.sqlite_path = self.output_dir / cfg.sqlite_filename
        self.metadata_path = self.output_dir / cfg.metadata_filename

        self._db: sqlite3.Connection | None = None
        self._sim_buffer: list[tuple[float, ...]] = []
        self._context_buffer: list[tuple[Any, ...]] = []
        self._spatial_buffer: list[tuple[Any, ...]] = []
        self._joint_indices_by_side: dict[str, list[int]] = {}
        self._joint_names_by_side: dict[str, tuple[str, ...]] = {}
        self._body_indices_by_side: dict[str, list[int]] = {}
        self._body_names_by_side: dict[str, tuple[str, ...]] = {}
        self._sim_columns: list[str] = []
        self._spatial_columns: list[str] = []
        self._row_count = 0
        self._initialized = False
        self._closed = False
        self._sim_dt: float | None = None
        self._context_metadata: dict[str, Any] = {}

    @property
    def num_dofs(self) -> int:
        return len(self._sim_columns) // 4

    def initialize(self, robot, *, sim_dt: float | None = None, metadata: dict[str, Any] | None = None) -> None:
        """Resolve runtime indices and create output tables."""

        if self._initialized:
            raise RuntimeError("DataRecording is already initialized.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.cfg.overwrite:
            existing = [path for path in (self.sqlite_path, self.metadata_path) if path.exists()]
            if existing:
                raise FileExistsError(f"Recording output already exists: {existing}")
        for path in (self.sqlite_path, self.metadata_path):
            if self.cfg.overwrite and path.exists():
                path.unlink()

        self._sim_dt = sim_dt
        self._context_metadata = dict(metadata or {})
        self._resolve_joint_indices(robot)
        self._resolve_body_indices(robot)
        self._create_tables()
        self._initialized = True

    def record_step(
        self,
        *,
        step_index: int,
        sim_time: float,
        robot,
        extra_context: dict[str, Any] | None = None,
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
        tau_all = self._tau_tensor(robot)
        context = dict(extra_context or {})

        for env_id in range(robot.num_instances):
            for side in self._selected_sides():
                joint_indices = self._joint_indices_by_side[side]
                q = q_all[env_id, joint_indices].detach().cpu().tolist()
                dq = dq_all[env_id, joint_indices].detach().cpu().tolist()
                ddq = ddq_all[env_id, joint_indices].detach().cpu().tolist()
                tau = tau_all[env_id, joint_indices].detach().cpu().tolist()

                sample_id = self._row_count
                self._sim_buffer.append(tuple(float(value) for value in [*q, *dq, *ddq, *tau]))
                self._context_buffer.append((sample_id, int(step_index), float(sim_time), int(env_id), side))
                if self.cfg.record_spatial_state:
                    self._record_spatial_rows(sample_id, step_index, sim_time, env_id, side, robot)
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
        if self._context_buffer:
            self._db.executemany(
                f"INSERT INTO {_quote_identifier(self.cfg.context_table_name)} VALUES (?, ?, ?, ?, ?)",
                self._context_buffer,
            )
            self._context_buffer.clear()
        if self._spatial_buffer:
            placeholders = ", ".join("?" for _ in self._spatial_columns)
            columns = ", ".join(_quote_identifier(name) for name in self._spatial_columns)
            self._db.executemany(
                f"INSERT INTO {_quote_identifier(self.cfg.spatial_table_name)} ({columns}) VALUES ({placeholders})",
                self._spatial_buffer,
            )
            self._spatial_buffer.clear()
        self._db.commit()

    def close(self) -> None:
        """Flush rows, write metadata, and close the SQLite connection."""

        if self._closed:
            return
        self.flush()
        self._write_metadata()
        if self._db is not None:
            self._db.close()
            self._db = None
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
        if self.cfg.joint_set != "tendon_chain_5":
            raise ValueError(f"Unknown joint_set: {self.cfg.joint_set!r}")
        return TENDON_CHAIN_5_JOINTS[side]

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

    def _create_tables(self) -> None:
        self._db = sqlite3.connect(self.sqlite_path)
        sim_columns_sql = ", ".join(f"{_quote_identifier(name)} REAL NOT NULL" for name in self._sim_columns)
        self._db.execute(f"CREATE TABLE {_quote_identifier(self.cfg.sim_table_name)} ({sim_columns_sql})")
        self._db.execute(
            f"""
            CREATE TABLE {_quote_identifier(self.cfg.context_table_name)} (
                sample_id INTEGER PRIMARY KEY,
                step_index INTEGER NOT NULL,
                time REAL NOT NULL,
                env_id INTEGER NOT NULL,
                side TEXT NOT NULL
            )
            """
        )
        if self.cfg.record_spatial_state:
            self._spatial_columns = _spatial_columns()
            spatial_sql = ", ".join(_spatial_column_sql(name) for name in self._spatial_columns)
            self._db.execute(f"CREATE TABLE {_quote_identifier(self.cfg.spatial_table_name)} ({spatial_sql})")
        self._db.commit()

    def _tau_tensor(self, robot):
        if self.cfg.tau_source == "applied_torque":
            return robot.data.applied_torque
        if self.cfg.tau_source == "computed_torque":
            return robot.data.computed_torque
        if self.cfg.tau_source == "zero":
            return robot.data.joint_pos * 0.0
        raise ValueError(f"Unsupported tau_source: {self.cfg.tau_source!r}")

    def _record_spatial_rows(
        self,
        sample_id: int,
        step_index: int,
        sim_time: float,
        env_id: int,
        side: str,
        robot,
    ) -> None:
        body_indices = self._body_indices_by_side.get(side)
        if not body_indices:
            return

        root_state = robot.data.root_state_w[env_id].detach().cpu().tolist()
        body_link_state = robot.data.body_link_state_w[env_id, body_indices].detach().cpu().tolist()
        body_com_state = robot.data.body_com_state_w[env_id, body_indices].detach().cpu().tolist()
        body_com_acc = robot.data.body_com_acc_w[env_id, body_indices].detach().cpu().tolist()

        for local_index, body_index in enumerate(body_indices):
            body_name = self._body_names_by_side[side][local_index]
            row = (
                int(sample_id),
                int(step_index),
                float(sim_time),
                int(env_id),
                side,
                int(body_index),
                body_name,
                *[float(value) for value in root_state],
                *[float(value) for value in body_link_state[local_index]],
                *[float(value) for value in body_com_state[local_index]],
                *[float(value) for value in body_com_acc[local_index]],
            )
            self._spatial_buffer.append(row)

    def _write_metadata(self) -> None:
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sqlite_path": str(self.sqlite_path),
            "sim_table_name": self.cfg.sim_table_name,
            "context_table_name": self.cfg.context_table_name,
            "spatial_table_name": self.cfg.spatial_table_name if self.cfg.record_spatial_state else None,
            "num_dofs": self.num_dofs,
            "row_count": self._row_count,
            "sim_columns": self._sim_columns,
            "sim_dt": self._sim_dt,
            "tau_source": self.cfg.tau_source,
            "available_tau_sources": ["applied_torque", "computed_torque", "zero"],
            "sim_units": {"q": "rad", "dq": "rad/s", "ddq": "rad/s^2", "tau": "N*m"},
            "joint_mappings": self._joint_metadata(),
            "body_mappings": self._body_metadata(),
            "omitted_first_pass_joint_names": {
                side: list(OMITTED_FIRST_PASS_JOINTS[side]) for side in self._selected_sides()
            },
            "config": _jsonable_config(self.cfg),
            "runtime_metadata": self._context_metadata,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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


def _jsonable_config(cfg: DataRecordingConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["output_dir"] = str(data["output_dir"])
    return data
