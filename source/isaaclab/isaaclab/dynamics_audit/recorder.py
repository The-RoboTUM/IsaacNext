# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SQLite recorder for simple dynamics audits."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import torch

from .candidates import print_residual_report, summarize_residual_candidates

VECTOR_TERM_NAMES = (
    "q",
    "dq",
    "ddq",
    "ddq_fd",
    "inertia",
    "inertia_fd",
    "gravity_force_api",
    "gravity_compensation",
    "gravity_compensation_actual",
    "coriolis_force_api",
    "coriolis_compensation",
    "coriolis_compensation_actual",
    "actuation_command",
    "actuation_previous_command",
    "applied_torque",
    "computed_torque",
    "joint_effort_target",
    "implicit_drive_estimate",
    "implicit_drive_saturation",
    "physx_actuation",
    "solver_joint",
    "contact",
    "friction",
    "residual_selected",
)


class DynamicsAuditRecorder:
    """Record and summarize generic articulation dynamics audit rows."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        asset_name: str,
        coordinate_names: list[str],
        num_envs: int,
        include_mass_matrix: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_name = str(asset_name)
        self.coordinate_names = tuple(str(name) for name in coordinate_names)
        self.num_dofs = len(self.coordinate_names)
        self.num_envs = int(num_envs)
        self.include_mass_matrix = bool(include_mass_matrix)
        self.metadata = dict(metadata or {})
        self.sqlite_path = self.output_dir / "dynamics_audit.db"
        self.metadata_path = self.output_dir / "metadata.json"
        if self.sqlite_path.exists():
            raise FileExistsError(f"Audit database already exists: {self.sqlite_path}. Pass --overwrite to replace it.")
        self._db = sqlite3.connect(self.sqlite_path)
        self._columns = self._make_columns()
        self._create_table()
        self._rows_for_summary: list[dict[str, tuple[float, ...]]] = []
        self._row_count = 0

    def record(self, *, step_index: int, time: float, terms: dict[str, torch.Tensor]) -> None:
        """Record all env rows for one simulation step."""

        rows = []
        for env_id in range(self.num_envs):
            row: list[Any] = [self._row_count, int(step_index), float(time), int(env_id)]
            summary_row: dict[str, tuple[float, ...]] = {}
            for term_name in VECTOR_TERM_NAMES:
                values = _vector_values(terms[term_name], env_id=env_id)
                row.extend(values)
                summary_row[term_name] = tuple(values)
            if self.include_mass_matrix:
                row.extend(_matrix_values(terms["mass_matrix"], env_id=env_id))
            rows.append(tuple(row))
            self._rows_for_summary.append(summary_row)
            self._row_count += 1
        placeholders = ", ".join("?" for _ in self._columns)
        columns = ", ".join(_quote_identifier(name) for name in self._columns)
        self._db.executemany(f"INSERT INTO audit_data ({columns}) VALUES ({placeholders})", rows)

    def close(self, *, print_report: bool = True) -> dict[str, Any]:
        """Finalize the database and write metadata."""

        self._db.commit()
        summary = summarize_residual_candidates(
            self._rows_for_summary,
            coordinate_names=self.coordinate_names,
            groups=self._coordinate_groups(),
        )
        full_metadata = {
            "asset_name": self.asset_name,
            "num_envs": self.num_envs,
            "num_dofs": self.num_dofs,
            "coordinate_names": list(self.coordinate_names),
            "sqlite_path": str(self.sqlite_path),
            "row_count": self._row_count,
            "terms": list(VECTOR_TERM_NAMES),
            "include_mass_matrix": self.include_mass_matrix,
            "equation": (
                "residual = lhs - rhs. Selected convention is "
                "Mddq - gravity_compensation_actual - coriolis_compensation_actual - actuation_command."
            ),
            "extra": self.metadata,
            "summary": summary,
        }
        self.metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
        self._db.close()
        if print_report:
            print_residual_report(summary)
            print(f"[INFO] Dynamics audit database saved to: {self.sqlite_path}")
            print(f"[INFO] Dynamics audit metadata saved to: {self.metadata_path}")
        return full_metadata

    def _create_table(self) -> None:
        columns_sql = ", ".join(_column_sql(name) for name in self._columns)
        self._db.execute(f"CREATE TABLE audit_data ({columns_sql})")
        self._db.execute("CREATE INDEX audit_data_step_idx ON audit_data (step_index, env_id)")
        self._db.commit()

    def _make_columns(self) -> list[str]:
        columns = ["sample_id", "step_index", "time", "env_id"]
        for term_name in VECTOR_TERM_NAMES:
            columns.extend(f"{term_name}{index}" for index in range(self.num_dofs))
        if self.include_mass_matrix:
            columns.extend(f"mass_matrix{row}_{col}" for row in range(self.num_dofs) for col in range(self.num_dofs))
        return columns

    def _coordinate_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {"all": list(range(self.num_dofs))}
        cart = [index for index, name in enumerate(self.coordinate_names) if "cart" in name or "slider" in name]
        pole = [index for index, name in enumerate(self.coordinate_names) if "pole" in name and "pendulum" not in name]
        pendulum = [index for index, name in enumerate(self.coordinate_names) if "pendulum" in name]
        if cart:
            groups["cart"] = cart
        if pole:
            groups["pole"] = pole
        if pendulum:
            groups["pendulum"] = pendulum
        return groups


def _vector_values(values: torch.Tensor, *, env_id: int) -> list[float]:
    return [float(value) for value in values[int(env_id)].detach().cpu().reshape(-1).tolist()]


def _matrix_values(values: torch.Tensor, *, env_id: int) -> list[float]:
    return [float(value) for value in values[int(env_id)].detach().cpu().reshape(-1).tolist()]


def _column_sql(name: str) -> str:
    if name in ("sample_id", "step_index", "env_id"):
        return f"{_quote_identifier(name)} INTEGER NOT NULL"
    return f"{_quote_identifier(name)} REAL NOT NULL"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
