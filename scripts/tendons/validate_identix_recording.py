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
from pathlib import Path
from typing import Any

ACTUATED_JOINT_NAMES = (
    "l0_acetabulofemoral_roll",
    "l1_acetabulofemoral_lateral",
    "l2_pseudo_acetabulofemoral_flexion",
    "r0_acetabulofemoral_roll",
    "r1_acetabulofemoral_lateral",
    "r2_pseudo_acetabulofemoral_flexion",
    "l8_knee_flexor",
    "r8_knee_flexor",
)

FORCE_BALANCE_TOLERANCE = 1.0e-3


class ValidationError(RuntimeError):
    """Raised when a recording violates the expected schema or data contract."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Recording output directory or SQLite database path.")
    parser.add_argument("--metadata", type=str, default=None, help="Metadata JSON path. Defaults to metadata.json.")
    parser.add_argument(
        "--num_dofs",
        type=int,
        default=None,
        help="Expected number of generalized coordinates. Defaults to metadata num_dofs.",
    )
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


def expected_dynamics_columns(num_dofs: int) -> list[str]:
    terms = ("inertia", "coriolis", "gravity", "tendon", "actuation", "contact", "friction", "external")
    return ["sample_id", "step_index", "time", "env_id", "side"] + [
        f"tau_{term}{i}" for term in terms for i in range(num_dofs)
    ]


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def resolve_paths(recording: Path, metadata_arg: str | None) -> tuple[Path, Path, dict[str, Any]]:
    if recording.is_dir():
        metadata_path = Path(metadata_arg) if metadata_arg is not None else recording / "metadata.json"
        metadata = load_metadata(metadata_path)
        sqlite_name = Path(str(metadata.get("sqlite_path", "forrest_kinematics.db"))).name
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


def resolve_dynamics_path(recording: Path, metadata: dict[str, Any]) -> Path | None:
    dynamics_path = metadata.get("dynamics_sqlite_path")
    if dynamics_path is None:
        return None
    if recording.is_dir():
        return recording / Path(str(dynamics_path)).name
    return recording.with_name(Path(str(dynamics_path)).name)


def resolve_debug_path(recording: Path, metadata: dict[str, Any]) -> Path | None:
    debug_path = metadata.get("debug_sqlite_path")
    if debug_path is None:
        return None
    if recording.is_dir():
        return recording / Path(str(debug_path)).name
    return recording.with_name(Path(str(debug_path)).name)


def validate_dynamics_db(path: Path, num_dofs: int, expected_rows: int) -> str:
    if not path.exists():
        raise ValidationError(f"Dynamics database is listed in metadata but does not exist: {path}")

    expected_columns = expected_dynamics_columns(num_dofs)
    with sqlite3.connect(path) as db:
        if not table_exists(db, "dynamics_data"):
            raise ValidationError(f"Missing dynamics_data table in {path}")
        actual_columns = table_columns(db, "dynamics_data")
        if actual_columns != expected_columns:
            raise ValidationError(
                f"Unexpected dynamics_data columns.\nExpected: {expected_columns}\nActual:   {actual_columns}"
            )
        select_columns = ", ".join(quote_identifier(name) for name in expected_columns)
        rows = db.execute(f"SELECT {select_columns} FROM dynamics_data ORDER BY sample_id").fetchall()

    if len(rows) != expected_rows:
        raise ValidationError(f"dynamics_data has {len(rows)} rows; expected {expected_rows}.")
    numeric_rows = []
    for row in rows:
        numeric_rows.append(tuple(float(value) for value in row if not isinstance(value, str)))
    require_finite(numeric_rows, "dynamics_data")
    col_index = {name: index for index, name in enumerate(expected_columns)}
    max_external_error = 0.0
    for row in rows:
        for dof_index in range(num_dofs):
            external = float(row[col_index[f"tau_external{dof_index}"]])
            components = (
                float(row[col_index[f"tau_actuation{dof_index}"]])
                + float(row[col_index[f"tau_contact{dof_index}"]])
                + float(row[col_index[f"tau_friction{dof_index}"]])
            )
            max_external_error = max(max_external_error, abs(external - components))
    if max_external_error > 1.0e-4:
        raise ValidationError(
            f"dynamics_data violates tau_external = tau_actuation + tau_contact + tau_friction: "
            f"max error {max_external_error:.6e}"
        )

    return f"Validated dynamics: {path} ({len(rows)} rows, external max error {max_external_error:.3e})"


def load_dynamics_streams(
    path: Path, sim_rows: list[tuple[int, tuple[float, ...]]]
) -> list[list[tuple[int, tuple[float, ...]]]]:
    if not path.exists():
        return [sim_rows]
    sample_by_id = {sample_id: row for sample_id, row in sim_rows}
    streams: dict[tuple[int, str], list[tuple[int, tuple[float, ...]]]] = {}
    with sqlite3.connect(path) as db:
        if not table_exists(db, "dynamics_data"):
            return [sim_rows]
        rows = db.execute(
            "SELECT sample_id, step_index, env_id, side FROM dynamics_data ORDER BY env_id, side, step_index"
        ).fetchall()
    for sample_id, step_index, env_id, side in rows:
        sample_id = int(sample_id)
        if sample_id not in sample_by_id:
            raise ValidationError(f"dynamics_data sample_id {sample_id} is missing from sim_data.")
        streams.setdefault((int(env_id), str(side)), []).append((int(step_index), sample_by_id[sample_id]))
    return list(streams.values()) or [sim_rows]


def validate_debug_db(path: Path, metadata: dict[str, Any], num_dofs: int) -> str | None:
    if int(metadata.get("debug_row_count", 0)) == 0:
        return None
    if not path.exists():
        raise ValidationError(f"Debug database is listed in metadata but does not exist: {path}")

    table_name = metadata.get("debug_table_name", "debug_data")
    with sqlite3.connect(path) as db:
        if not table_exists(db, table_name):
            raise ValidationError(f"Missing {table_name} table in {path}")
        columns = table_columns(db, table_name)
        terms = (
            "inertia",
            "gravity",
            "coriolis",
            "tendon",
            "actuation_command",
            "contact_validated",
            "friction",
            "residual",
        )
        required = [f"tau_{term}{dof_index}" for term in terms for dof_index in range(num_dofs)]
        missing = [column for column in required if column not in columns]
        if missing:
            raise ValidationError(f"Debug database is missing force-balance columns: {missing}")
        pantograph_columns = [
            f"tau_{term}{dof_index}"
            for term in (
                "pantograph_spring",
                "pantograph_damping",
                "pantograph_actuation",
                "pantograph_applied_actuation",
                "pantograph_computed_actuation",
                "pantograph_reconstructed_actuation",
                "pantograph_actuation_error",
            )
            for dof_index in range(num_dofs)
        ]
        selected_columns = required + [column for column in pantograph_columns if column in columns]
        select_columns = ", ".join(quote_identifier(name) for name in selected_columns)
        rows = db.execute(f"SELECT {select_columns} FROM {quote_identifier(table_name)} ORDER BY sample_id").fetchall()

    require_finite([tuple(float(value) for value in row) for row in rows], table_name)
    col_index = {name: index for index, name in enumerate(selected_columns)}
    max_residual_error = 0.0
    for row in rows:
        for dof_index in range(num_dofs):
            conservative = (
                float(row[col_index[f"tau_inertia{dof_index}"]])
                + float(row[col_index[f"tau_gravity{dof_index}"]])
                + float(row[col_index[f"tau_coriolis{dof_index}"]])
                + float(row[col_index[f"tau_tendon{dof_index}"]])
            )
            non_conservative = (
                float(row[col_index[f"tau_actuation_command{dof_index}"]])
                + float(row[col_index[f"tau_contact_validated{dof_index}"]])
                + float(row[col_index[f"tau_friction{dof_index}"]])
            )
            residual = float(row[col_index[f"tau_residual{dof_index}"]])
            max_residual_error = max(max_residual_error, abs(residual - (conservative - non_conservative)))
    if max_residual_error > FORCE_BALANCE_TOLERANCE:
        raise ValidationError(
            "Debug database violates residual = "
            "inertia + gravity + coriolis + tendon - actuation - contact - friction: "
            f"max error {max_residual_error:.6e}"
        )

    max_pantograph_leakage = max(
        validate_pantograph_columns(rows, selected_columns, metadata, num_dofs, "pantograph_spring"),
        validate_pantograph_columns(rows, selected_columns, metadata, num_dofs, "pantograph_damping"),
        validate_pantograph_columns(rows, selected_columns, metadata, num_dofs, "pantograph_actuation"),
        validate_pantograph_columns(rows, selected_columns, metadata, num_dofs, "pantograph_applied_actuation"),
        validate_pantograph_columns(rows, selected_columns, metadata, num_dofs, "pantograph_computed_actuation"),
        validate_pantograph_columns(rows, selected_columns, metadata, num_dofs, "pantograph_reconstructed_actuation"),
        validate_pantograph_columns(rows, selected_columns, metadata, num_dofs, "pantograph_actuation_error"),
    )

    return (
        f"Validated debug force balance: {path} ({len(rows)} rows, residual max error {max_residual_error:.3e}, "
        f"pantograph leakage {max_pantograph_leakage:.3e})"
    )


def validate_pantograph_columns(
    rows: list[tuple[Any, ...]],
    columns: list[str],
    metadata: dict[str, Any],
    num_dofs: int,
    term_name: str,
) -> float:
    term_columns = [f"tau_{term_name}{dof_index}" for dof_index in range(num_dofs)]
    if any(column not in columns for column in term_columns):
        return 0.0

    mappings = sorted(metadata.get("joint_mappings", []), key=lambda item: int(item.get("q_index", -1)))
    if len(mappings) != num_dofs:
        raise ValidationError("Metadata joint_mappings length does not match num_dofs.")

    col_index = {name: index for index, name in enumerate(columns)}
    max_leakage = 0.0
    for mapping in mappings:
        q_index = int(mapping["q_index"])
        joint_name = str(mapping["joint_name"])
        if joint_name in ("lp1_pantograph", "rp1_pantograph"):
            continue
        column = f"tau_{term_name}{q_index}"
        max_leakage = max(max_leakage, max(abs(float(row[col_index[column]])) for row in rows))
    if max_leakage > 1.0e-9:
        raise ValidationError(f"{term_name} has nonzero values on non-pantograph joints: {max_leakage:.6e}")
    return max_leakage


def require_finite(rows: list[tuple[float, ...]], label: str) -> None:
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            if not math.isfinite(value):
                raise ValidationError(f"{label} contains non-finite value at row {row_index}, column {col_index}.")


def validate_motor_tau(rows: list[tuple[float, ...]], metadata: dict[str, Any], num_dofs: int) -> None:
    if metadata.get("tau_source") != "motor_torque":
        return
    motor_names = set(ACTUATED_JOINT_NAMES)
    mappings = sorted(metadata.get("joint_mappings", []), key=lambda item: int(item.get("q_index", -1)))
    if len(mappings) != num_dofs:
        raise ValidationError("Metadata joint_mappings length does not match num_dofs.")

    tau_offset = 3 * num_dofs
    for mapping in mappings:
        q_index = int(mapping["q_index"])
        joint_name = str(mapping["joint_name"])
        if joint_name in motor_names:
            continue
        max_abs_tau = max(abs(row[tau_offset + q_index]) for row in rows)
        if max_abs_tau > 1.0e-9:
            raise ValidationError(
                f"motor_torque tau has nonzero value on non-motor joint {joint_name} (q{q_index}): {max_abs_tau:.6e}"
            )


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
    streams: list[list[tuple[int, tuple[float, ...]]]],
    num_dofs: int,
    sample_dt: float,
) -> dict[str, tuple[float, float, int] | None]:
    return {
        "q_to_dq": derivative_residual(streams, 0, num_dofs, num_dofs, sample_dt),
        "dq_to_ddq": derivative_residual(streams, num_dofs, 2 * num_dofs, num_dofs, sample_dt),
    }


def derivative_residual(
    streams: list[list[tuple[int, tuple[float, ...]]]],
    value_offset: int,
    derivative_offset: int,
    num_dofs: int,
    sample_dt: float,
):
    residuals = []
    if sample_dt <= 0.0:
        return None
    central_dt = 2.0 * sample_dt
    for sim_rows in streams:
        if len(sim_rows) < 3:
            continue
        for index in range(1, len(sim_rows) - 1):
            step_prev, values_prev = sim_rows[index - 1]
            step, values = sim_rows[index]
            step_next, values_next = sim_rows[index + 1]
            if step - step_prev != 1 or step_next - step != 1:
                continue
            for dof_index in range(num_dofs):
                estimated = (values_next[value_offset + dof_index] - values_prev[value_offset + dof_index]) / central_dt
                actual = values[derivative_offset + dof_index]
                residuals.append(actual - estimated)

    if not residuals:
        return None
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    max_abs = max(abs(value) for value in residuals)
    return rms, max_abs, len(residuals)


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
    num_dofs = args.num_dofs if args.num_dofs is not None else int(metadata.get("num_dofs", 5))
    expected_columns = expected_sim_columns(num_dofs)

    with sqlite3.connect(sqlite_path) as db:
        if not table_exists(db, table_name):
            raise ValidationError(f"Missing sim_data table: {table_name}")
        data_tables = [
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
        ]
        if data_tables != [table_name]:
            raise ValidationError(f"Expected only {table_name!r} table in database; found {data_tables}.")

        actual_columns = table_columns(db, table_name)
        if actual_columns != expected_columns:
            raise ValidationError(
                f"Unexpected sim_data columns.\nExpected: {expected_columns}\nActual:   {actual_columns}"
            )

        sim_rows = load_sim_rows(db, table_name, expected_columns)
        if not sim_rows:
            raise ValidationError("sim_data table is empty.")

        values = [row_values for _, row_values in sim_rows]
        require_finite(values, "sim_data")
        validate_motor_tau(values, metadata, num_dofs)
        sim_dt = float(metadata.get("sim_dt", 0.0))
        sampling_stride = int(metadata.get("config", {}).get("sampling_stride", 1))
        stats = compute_stats(values, expected_columns)

    if int(metadata.get("num_dofs", -1)) != num_dofs:
        raise ValidationError(f"Metadata num_dofs={metadata.get('num_dofs')} does not match expected {num_dofs}.")
    if metadata.get("sim_columns") != expected_columns:
        raise ValidationError("Metadata sim_columns do not match the SQLite sim_data schema.")
    if int(metadata.get("row_count", -1)) != len(sim_rows):
        raise ValidationError(f"Metadata row_count={metadata.get('row_count')} does not match {len(sim_rows)} rows.")
    if metadata.get("dynamics_sqlite_path") is not None:
        expected_dynamics_rows = int(metadata.get("dynamics_row_count", -1))
        if expected_dynamics_rows != len(sim_rows):
            raise ValidationError(
                f"Metadata dynamics_row_count={expected_dynamics_rows} does not match {len(sim_rows)} sim rows."
            )
    expected_units = {"q": "rad", "dq": "rad/s", "ddq": "rad/s^2", "tau": "N*m"}
    if metadata.get("sim_units") != expected_units:
        raise ValidationError(f"Metadata sim_units must be {expected_units}; got {metadata.get('sim_units')}.")
    for mapping in metadata.get("joint_mappings", []):
        if mapping.get("units") != "rad":
            raise ValidationError(f"Joint mapping has non-radian units: {mapping}")

    dynamics_report = None
    dynamics_path = resolve_dynamics_path(Path(args.recording), metadata)
    if dynamics_path is not None:
        dynamics_report = validate_dynamics_db(dynamics_path, num_dofs, len(sim_rows))
        fd_streams = load_dynamics_streams(dynamics_path, sim_rows)
    else:
        fd_streams = [sim_rows]
    sim_dt = float(metadata.get("sim_dt", 0.0))
    sampling_stride = int(metadata.get("config", {}).get("sampling_stride", 1))
    fd_report = finite_difference_report(fd_streams, num_dofs, sim_dt * sampling_stride)
    debug_report = None
    debug_path = resolve_debug_path(Path(args.recording), metadata)
    if debug_path is not None:
        debug_report = validate_debug_db(debug_path, metadata, num_dofs)

    print(f"Validated recording: {sqlite_path}")
    print(f"Metadata: {metadata_path}")
    print(f"sim_data rows: {len(sim_rows)}")
    if dynamics_report is not None:
        print(dynamics_report)
    if debug_report is not None:
        print(debug_report)

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
        print(check_identix_loader(sqlite_path, table_name, num_dofs, args.identix_repo))


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
