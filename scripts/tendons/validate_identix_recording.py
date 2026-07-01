# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate a tiny Identix-style Forrest tendon recording."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    """Raised when a recording violates the expected schema or data contract."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Recording output directory or SQLite database path.")
    parser.add_argument("--metadata", type=str, default=None, help="Metadata JSON path. Defaults to metadata.json.")
    parser.add_argument("--num_dofs", type=int, default=5, help="Expected number of generalized coordinates.")
    parser.add_argument("--table_name", type=str, default=None, help="Override the sim_data table name.")
    parser.add_argument(
        "--max_fd_error",
        type=float,
        default=None,
        help="Optional max allowed central-difference residual for q->dq and dq->ddq.",
    )
    parser.add_argument(
        "--check_identix",
        action="store_true",
        help="Also load the database through identix.data_manager.SystemDataset.",
    )
    parser.add_argument(
        "--identix_repo",
        type=str,
        default=None,
        help="Identix repo path used when --check_identix is set.",
    )
    return parser.parse_args()


def expected_sim_columns(num_dofs: int) -> list[str]:
    return (
        [f"q{i}" for i in range(num_dofs)]
        + [f"dq{i}" for i in range(num_dofs)]
        + [f"ddq{i}" for i in range(num_dofs)]
        + [f"tau{i}" for i in range(num_dofs)]
    )


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def resolve_paths(recording: Path, metadata_arg: str | None) -> tuple[Path, Path, dict[str, Any]]:
    if recording.is_dir():
        metadata_path = Path(metadata_arg) if metadata_arg is not None else recording / "metadata.json"
        metadata = load_metadata(metadata_path)
        sqlite_name = Path(str(metadata.get("sqlite_path", "forrest_tendon_chain_sim_data.db"))).name
        sqlite_path = recording / sqlite_name
        return sqlite_path, metadata_path, metadata

    metadata_path = Path(metadata_arg) if metadata_arg is not None else recording.with_name("metadata.json")
    metadata = load_metadata(metadata_path)
    return recording, metadata_path, metadata


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"Metadata file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def table_exists(db: sqlite3.Connection, table_name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(db: sqlite3.Connection, table_name: str) -> list[str]:
    rows = db.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    return [row[1] for row in rows]


def load_sim_rows(db: sqlite3.Connection, table_name: str, columns: list[str]) -> list[tuple[int, tuple[float, ...]]]:
    select_columns = ", ".join(quote_identifier(name) for name in columns)
    rows = db.execute(f"SELECT rowid, {select_columns} FROM {quote_identifier(table_name)} ORDER BY rowid").fetchall()
    return [(int(row[0]) - 1, tuple(float(value) for value in row[1:])) for row in rows]


def load_context_rows(db: sqlite3.Connection, table_name: str) -> list[tuple[int, int, float, int, str]]:
    rows = db.execute(
        f"SELECT sample_id, step_index, time, env_id, side FROM {quote_identifier(table_name)} ORDER BY sample_id"
    ).fetchall()
    return [
        (int(sample_id), int(step), float(time), int(env_id), str(side)) for sample_id, step, time, env_id, side in rows
    ]


def require_finite(rows: list[tuple[float, ...]], label: str) -> None:
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            if not math.isfinite(value):
                raise ValidationError(f"{label} contains non-finite value at row {row_index}, column {col_index}.")


def compute_stats(rows: list[tuple[float, ...]], columns: list[str]) -> list[tuple[str, float, float, float, float]]:
    if not rows:
        return []
    stats = []
    for col_index, column in enumerate(columns):
        values = [row[col_index] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stats.append((column, min(values), max(values), mean, math.sqrt(variance)))
    return stats


def finite_difference_report(
    sim_rows: list[tuple[int, tuple[float, ...]]],
    context_rows: list[tuple[int, int, float, int, str]],
    num_dofs: int,
) -> dict[str, tuple[float, float, int] | None]:
    records_by_group = defaultdict(list)
    for (sample_id, values), context in zip(sim_rows, context_rows):
        context_sample_id, _, time, env_id, side = context
        if sample_id != context_sample_id:
            raise ValidationError(f"sim_data row order does not match sample_context at sample_id {context_sample_id}.")
        records_by_group[(env_id, side)].append((time, values))

    return {
        "q_to_dq": derivative_residual(records_by_group, 0, num_dofs, num_dofs),
        "dq_to_ddq": derivative_residual(records_by_group, num_dofs, 2 * num_dofs, num_dofs),
    }


def derivative_residual(records_by_group, value_offset: int, derivative_offset: int, num_dofs: int):
    residuals = []
    for records in records_by_group.values():
        records = sorted(records, key=lambda item: item[0])
        if len(records) < 3:
            continue
        for index in range(1, len(records) - 1):
            t_prev, values_prev = records[index - 1]
            t_next, values_next = records[index + 1]
            _, values = records[index]
            dt = t_next - t_prev
            if dt <= 0.0:
                continue
            for dof_index in range(num_dofs):
                estimated = (values_next[value_offset + dof_index] - values_prev[value_offset + dof_index]) / dt
                actual = values[derivative_offset + dof_index]
                residuals.append(actual - estimated)

    if not residuals:
        return None
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    max_abs = max(abs(value) for value in residuals)
    return rms, max_abs, len(residuals)


def validate_spatial_table(db: sqlite3.Connection, metadata: dict[str, Any]) -> int | None:
    spatial_table = metadata.get("spatial_table_name")
    if spatial_table is None:
        return None
    if not table_exists(db, spatial_table):
        raise ValidationError(f"Metadata references missing spatial table: {spatial_table}")

    columns = table_columns(db, spatial_table)
    required_columns = {"root_pos_z", "body_link_pos_z", "body_com_pos_z"}
    missing = sorted(required_columns.difference(columns))
    if missing:
        raise ValidationError(f"Spatial table is missing 3D diagnostic columns: {missing}")

    rows = db.execute(f"SELECT * FROM {quote_identifier(spatial_table)}").fetchall()
    if not rows:
        raise ValidationError("Spatial table exists but has no rows.")
    for row_index, row in enumerate(rows):
        for value in row:
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValidationError(f"Spatial table contains non-finite numeric value at row {row_index}.")
    return len(rows)


def check_identix_loader(sqlite_path: Path, table_name: str, num_dofs: int, identix_repo: str | None) -> str:
    repo_path = Path(identix_repo) if identix_repo else Path("/home/humanoid/repos/Identix")
    if repo_path.exists():
        sys.path.insert(0, str(repo_path / "src"))

    try:
        from identix.data_manager import SystemDataset
    except Exception as exc:
        raise ValidationError(
            "Could not import identix.data_manager.SystemDataset. Run --check_identix from an Identix environment "
            f"with its dependencies installed. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    dataset = SystemDataset(str(sqlite_path), num_dofs=num_dofs, table_name=table_name)
    state, label = dataset[0]
    return f"len={len(dataset)}, state_shape={tuple(state.shape)}, label_shape={tuple(label.shape)}"


def validate(args) -> None:
    sqlite_path, metadata_path, metadata = resolve_paths(Path(args.recording), args.metadata)
    if not sqlite_path.exists():
        raise ValidationError(f"SQLite database does not exist: {sqlite_path}")

    table_name = args.table_name or metadata.get("sim_table_name", "sim_data")
    context_table = metadata.get("context_table_name", "sample_context")
    expected_columns = expected_sim_columns(args.num_dofs)

    with sqlite3.connect(sqlite_path) as db:
        if not table_exists(db, table_name):
            raise ValidationError(f"Missing sim_data table: {table_name}")
        if not table_exists(db, context_table):
            raise ValidationError(f"Missing sample context table: {context_table}")

        actual_columns = table_columns(db, table_name)
        if actual_columns != expected_columns:
            raise ValidationError(
                f"Unexpected sim_data columns.\nExpected: {expected_columns}\nActual:   {actual_columns}"
            )

        sim_rows = load_sim_rows(db, table_name, expected_columns)
        if not sim_rows:
            raise ValidationError("sim_data table is empty.")
        context_rows = load_context_rows(db, context_table)
        if len(context_rows) != len(sim_rows):
            raise ValidationError(
                f"Context row count {len(context_rows)} does not match sim_data rows {len(sim_rows)}."
            )

        values = [row_values for _, row_values in sim_rows]
        require_finite(values, "sim_data")
        fd_report = finite_difference_report(sim_rows, context_rows, args.num_dofs)
        spatial_rows = validate_spatial_table(db, metadata)
        stats = compute_stats(values, expected_columns)

    if int(metadata.get("num_dofs", -1)) != args.num_dofs:
        raise ValidationError(f"Metadata num_dofs={metadata.get('num_dofs')} does not match expected {args.num_dofs}.")
    if metadata.get("sim_columns") != expected_columns:
        raise ValidationError("Metadata sim_columns do not match the SQLite sim_data schema.")
    if int(metadata.get("row_count", -1)) != len(sim_rows):
        raise ValidationError(f"Metadata row_count={metadata.get('row_count')} does not match {len(sim_rows)} rows.")
    expected_units = {"q": "rad", "dq": "rad/s", "ddq": "rad/s^2", "tau": "N*m"}
    if metadata.get("sim_units") != expected_units:
        raise ValidationError(f"Metadata sim_units must be {expected_units}; got {metadata.get('sim_units')}.")
    for mapping in metadata.get("joint_mappings", []):
        if mapping.get("units") != "rad":
            raise ValidationError(f"Joint mapping has non-radian units: {mapping}")

    print(f"Validated recording: {sqlite_path}")
    print(f"Metadata: {metadata_path}")
    print(f"sim_data rows: {len(sim_rows)}")
    print(f"context rows: {len(context_rows)}")
    if spatial_rows is not None:
        print(f"spatial rows: {spatial_rows}")

    for label, report in fd_report.items():
        if report is None:
            print(f"{label}: not enough samples for central differences")
            continue
        rms, max_abs, count = report
        print(f"{label}: rms={rms:.6e}, max_abs={max_abs:.6e}, residuals={count}")
        if args.max_fd_error is not None and max_abs > args.max_fd_error:
            raise ValidationError(f"{label} max residual {max_abs:.6e} exceeds {args.max_fd_error:.6e}.")

    print("\nColumn statistics:")
    for column, min_value, max_value, mean, std in stats:
        print(f"{column:>5s}: min={min_value: .6e} max={max_value: .6e} mean={mean: .6e} std={std: .6e}")

    if args.check_identix:
        print("\nIdentix SystemDataset:")
        print(check_identix_loader(sqlite_path, table_name, args.num_dofs, args.identix_repo))


def main() -> int:
    args = parse_args()
    try:
        validate(args)
    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
