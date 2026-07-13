# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a sysid-ready Forrest recording from a debug dynamics run.

The selected final convention is the audited ``current`` baseline:

* inertia uses the same raw PhysX acceleration source exported in ``sim_data.ddq``;
* Coriolis, tendon, actuation, measured contact, and friction come directly from
  the recorded debug terms;
* gravity uses the recorded ``gravity_identification`` term, which keeps joint
  gravity from the PhysX force API and adds the missing floating-base gravity
  wrench inferred from the compensation API.

This produces an Identix-compatible ``sim_data`` database where ``tau`` is the
actuation generalized force only, plus a dynamics database with the decomposed
terms used to audit the full force balance.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SYSID_MODEL = "current"
DYNAMICS_TERMS = (
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


class ExportError(RuntimeError):
    """Raised when a recording cannot be converted to sysid format."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path, help="Recording directory containing metadata.json and debug.db.")
    parser.add_argument("--metadata", type=Path, default=None, help="Source metadata path. Defaults to metadata.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to recording_dir.")
    parser.add_argument("--kinematics-filename", default="forrest_sysid_kinematics.db")
    parser.add_argument("--dynamics-filename", default="forrest_sysid_dynamics.db")
    parser.add_argument("--metadata-filename", default="sysid_metadata.json")
    parser.add_argument(
        "--max-residual-norm",
        type=float,
        default=None,
        help="Optional residual-norm filter. Disabled by default; rows above this norm are excluded when set.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sysid export files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = export_sysid_recording(
            recording_dir=args.recording_dir,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            kinematics_filename=args.kinematics_filename,
            dynamics_filename=args.dynamics_filename,
            metadata_filename=args.metadata_filename,
            max_residual_norm=args.max_residual_norm,
            overwrite=args.overwrite,
        )
    except ExportError as exc:
        print(f"SYSID EXPORT FAILED: {exc}")
        return 1

    print(f"Sysid kinematics: {summary['kinematics_path']}")
    print(f"Sysid dynamics:   {summary['dynamics_path']}")
    print(f"Sysid metadata:   {summary['metadata_path']}")
    print(
        "Final balance residual: "
        f"mean={summary['residual_norm']['mean']:.3f}, "
        f"p95={summary['residual_norm']['p95']:.3f}, "
        f"max={summary['residual_norm']['max']:.3f}"
    )
    filter_summary = summary["residual_filter"]
    print(
        "Residual filter: "
        f"{'enabled' if filter_summary['enabled'] else 'disabled'}, "
        f"kept={filter_summary['kept_rows']}, rejected={filter_summary['rejected_rows']}"
    )
    thresholds = {item["label"]: item for item in summary.get("residual_cleanliness", {}).get("thresholds", [])}
    selected_thresholds = [
        thresholds[label]
        for label in ("absolute_10", "absolute_100", "absolute_300", "absolute_500")
        if label in thresholds
    ]
    if selected_thresholds:
        formatted = ", ".join(
            f">{item['threshold']:.0f}: {100.0 * item['fraction_above']:.2f}%" for item in selected_thresholds
        )
        print(f"Residual cleanliness: {formatted}")
    return 0


def export_sysid_recording(
    *,
    recording_dir: Path,
    metadata_path: Path | None,
    output_dir: Path | None,
    kinematics_filename: str,
    dynamics_filename: str,
    metadata_filename: str,
    max_residual_norm: float | None,
    overwrite: bool,
) -> dict[str, Any]:
    recording_dir = recording_dir.resolve()
    metadata_path = (metadata_path or recording_dir / "metadata.json").resolve()
    output_dir = (output_dir or recording_dir).resolve()
    metadata = _load_json(metadata_path)

    num_dofs = int(metadata["num_dofs"])
    sim_table_name = str(metadata.get("sim_table_name", "sim_data"))
    sim_columns = _sim_data_columns(num_dofs)
    dynamics_columns = _dynamics_columns(num_dofs)
    base_indices = _floating_base_indices(metadata)

    source_kinematics_path = _recording_file(recording_dir, metadata.get("sqlite_path", "forrest_kinematics.db"))
    source_debug_path = _recording_file(recording_dir, metadata.get("debug_sqlite_path", "debug.db"))
    if not source_kinematics_path.exists():
        raise ExportError(f"Missing source kinematics database: {source_kinematics_path}")
    if not source_debug_path.exists():
        raise ExportError(f"Missing source debug database: {source_debug_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    kinematics_path = output_dir / kinematics_filename
    dynamics_path = output_dir / dynamics_filename
    sysid_metadata_path = output_dir / metadata_filename
    for path in (kinematics_path, dynamics_path, sysid_metadata_path):
        if path.exists() and not overwrite:
            raise ExportError(f"Output already exists: {path}. Pass --overwrite to replace it.")
        if path.exists():
            path.unlink()

    sim_rows = _read_sim_rows(source_kinematics_path, sim_table_name, sim_columns)
    debug_rows, _ = _read_debug_rows(source_debug_path, num_dofs)
    if len(sim_rows) != len(debug_rows):
        raise ExportError(f"sim_data has {len(sim_rows)} rows but debug_data has {len(debug_rows)} rows.")
    kinematic_quality = _kinematic_quality_by_sample_id(sim_rows, debug_rows, num_dofs, metadata)

    sysid_sim_rows: list[tuple[float, ...]] = []
    sysid_dynamics_rows: list[tuple[Any, ...]] = []
    residual_norms: list[float] = []
    all_residual_records: list[dict[str, Any]] = []
    rejected_rows = 0
    for (sample_id, sim_values), debug_row in zip(sim_rows, debug_rows, strict=True):
        if int(debug_row["sample_id"]) != sample_id:
            raise ExportError(f"sample_id mismatch: sim row {sample_id}, debug row {debug_row['sample_id']}")
        terms = _sysid_terms(debug_row, num_dofs, base_indices)
        external = _vector_add(terms["actuation"], terms["contact"], terms["friction"])
        residual = _vector_sub(
            _vector_add(terms["inertia"], terms["gravity"], terms["coriolis"], terms["tendon"]),
            external,
        )
        residual_norm = _vector_norm(residual)
        all_residual_records.append(
            {
                "source_sample_id": sample_id,
                "step_index": int(debug_row["step_index"]),
                "env_id": int(debug_row["env_id"]),
                "side": str(debug_row["side"]),
                "residual_norm": residual_norm,
                **kinematic_quality.get(sample_id, {}),
            }
        )
        if max_residual_norm is not None and residual_norm > float(max_residual_norm):
            rejected_rows += 1
            continue
        residual_norms.append(residual_norm)

        q_dq_ddq = sim_values[: 3 * num_dofs]
        sysid_sim_rows.append((*q_dq_ddq, *terms["actuation"]))

        output_sample_id = len(sysid_dynamics_rows)
        dynamics_row: list[Any] = [
            output_sample_id,
            int(debug_row["step_index"]),
            float(debug_row["time"]),
            int(debug_row["env_id"]),
            str(debug_row["side"]),
        ]
        for term_name in DYNAMICS_TERMS:
            if term_name == "external":
                values = external
            elif term_name == "residual":
                values = residual
            else:
                values = terms[term_name]
            dynamics_row.extend(values)
        sysid_dynamics_rows.append(tuple(dynamics_row))

    _write_kinematics_db(kinematics_path, sim_table_name, sim_columns, sysid_sim_rows)
    _write_dynamics_db(dynamics_path, dynamics_columns, sysid_dynamics_rows)

    residual_filter = {
        "enabled": max_residual_norm is not None,
        "max_residual_norm": max_residual_norm,
        "input_rows": len(sim_rows),
        "kept_rows": len(sysid_sim_rows),
        "rejected_rows": rejected_rows,
        "rejected_fraction": rejected_rows / max(len(sim_rows), 1),
        "default_state": "disabled",
        "source_sample_id_policy": (
            "when filtering is enabled, output dynamics_data.sample_id is reindexed to match output sim_data rowid - 1"
        ),
    }
    coordinate_contract = _coordinate_contract(metadata, num_dofs)
    database_contract = _database_contract(
        metadata=metadata,
        sim_table_name=sim_table_name,
        sim_columns=sim_columns,
        dynamics_columns=dynamics_columns,
        coordinate_contract=coordinate_contract,
    )
    cleanliness = _residual_cleanliness_report(all_residual_records)
    summary = {
        "model": SYSID_MODEL,
        "rows": len(sysid_sim_rows),
        "base_indices": base_indices,
        "source_metadata": str(metadata_path),
        "source_kinematics": str(source_kinematics_path),
        "source_debug": str(source_debug_path),
        "kinematics_path": str(kinematics_path),
        "dynamics_path": str(dynamics_path),
        "metadata_path": str(sysid_metadata_path),
        "residual_norm": _norm_summary(residual_norms),
        "pre_filter_residual_norm": _norm_summary([record["residual_norm"] for record in all_residual_records]),
        "residual_filter": residual_filter,
        "residual_cleanliness": cleanliness,
        "sysid_model_policy": metadata.get("sysid_model_policy", {}),
        "tau_semantics": (
            "tau0..tauN in sim_data are actuation generalized forces only, following the Identix "
            "database convention. Contact, friction, and corrected gravity remain available in "
            "dynamics_data for the full force-balance residual."
        ),
        "dynamics_semantics": {
            "tau_gravity": (
                "recorded gravity_identification term: PhysX joint gravity from the force API plus the "
                "missing floating-base gravity wrench inferred from gravity_compensation_actual. This is "
                "exported directly from debug_data rather than recomputed during export."
            ),
            "tau_contact": "measured contact_validated generalized force, including accepted contact-point moment",
            "tau_external": (
                "tau_actuation + tau_contact + tau_friction; used by the residual, but not written "
                "to sim_data tau columns because Identix expects actuation only there"
            ),
            "tau_solver_constraint_internal": (
                "optional PhysX solver-projected passive/limit constraint reaction generalized force, kept "
                "separate from tau_external and sim_data tau so consumers can explicitly choose whether to model it"
            ),
            "equation": (
                "sysid convention: tau_inertia + tau_gravity + tau_coriolis + tau_tendon = tau_external + residual"
            ),
            "residual": (
                "residual = tau_inertia + tau_gravity + tau_coriolis + tau_tendon - tau_external; "
                "positive values mean the conservative/model side is larger than the measured external side"
            ),
        },
    }
    sysid_metadata = dict(metadata)
    sysid_metadata.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "sqlite_path": str(kinematics_path),
            "dynamics_sqlite_path": str(dynamics_path),
            "metadata_filename": metadata_filename,
            "debug_sqlite_path": str(source_debug_path),
            "row_count": len(sysid_sim_rows),
            "dynamics_row_count": len(sysid_dynamics_rows),
            "tau_source": "actuation_command",
            "ddq_source": metadata.get("ddq_source", "unknown"),
            "tau_semantics": summary["tau_semantics"],
            "ddq_semantics": metadata.get("ddq_semantics", ""),
            "dynamics_semantics": summary["dynamics_semantics"],
            "sysid_model_policy": summary["sysid_model_policy"],
            "identix_database_contract": database_contract,
            "dof_names": [coordinate["coordinate_name"] for coordinate in coordinate_contract],
            "dof_contract": coordinate_contract,
            "residual_filter": residual_filter,
            "residual_cleanliness": cleanliness,
            "runtime_metadata": {
                **dict(metadata.get("runtime_metadata", {})),
                "sysid_export": summary,
            },
        }
    )
    if isinstance(sysid_metadata.get("config"), dict):
        sysid_metadata["config"] = {
            **sysid_metadata["config"],
            "sqlite_filename": kinematics_filename,
            "dynamics_sqlite_filename": dynamics_filename,
            "metadata_filename": metadata_filename,
            "tau_source": "actuation_command",
            "ddq_source": metadata.get("ddq_source", "unknown"),
        }
    sysid_metadata_path.write_text(json.dumps(sysid_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _coordinate_contract(metadata: dict[str, Any], num_dofs: int) -> list[dict[str, Any]]:
    mappings = sorted(metadata.get("coordinate_mappings", []), key=lambda item: int(item.get("q_index", -1)))
    if len(mappings) != num_dofs:
        raise ExportError(f"Expected {num_dofs} coordinate mappings, found {len(mappings)}.")
    contract = []
    for mapping in mappings:
        q_index = int(mapping["q_index"])
        coordinate_type = str(mapping.get("coordinate_type", "joint"))
        coordinate_name = str(mapping.get("coordinate_name") or mapping.get("joint_name") or f"q{q_index}")
        row = {
            "q_index": q_index,
            "coordinate_name": coordinate_name,
            "coordinate_type": coordinate_type,
            "columns": {
                "q": f"q{q_index}",
                "dq": f"dq{q_index}",
                "ddq": f"ddq{q_index}",
                "tau": f"tau{q_index}",
            },
            "units": {
                "q": mapping.get("q_unit", mapping.get("units", "rad")),
                "dq": mapping.get("dq_unit", "rad/s"),
                "ddq": mapping.get("ddq_unit", "rad/s^2"),
                "tau": mapping.get("tau_unit", "N*m"),
            },
            "source_mapping": mapping,
        }
        if coordinate_type == "floating_base":
            row.update(_base_coordinate_contract(q_index, coordinate_name))
        else:
            row.update(
                {
                    "q_meaning": "Isaac joint position in the listed joint coordinate.",
                    "dq_meaning": "Isaac joint velocity in the listed joint coordinate.",
                    "ddq_meaning": "Recorded adjacent-step joint acceleration in the listed joint coordinate.",
                    "tau_meaning": (
                        "Generalized joint torque in N*m, positive in the Isaac joint coordinate direction. "
                        "In sysid sim_data this is tau_actuation for that coordinate."
                    ),
                    "force_frame": "joint_coordinate",
                    "positive_direction": "positive Isaac joint coordinate direction",
                    "sign_convention": mapping.get("sign_convention", "isaac_joint_position"),
                    "isaac_joint_index": mapping.get("isaac_joint_index"),
                }
            )
        contract.append(row)
    return contract


def _base_coordinate_contract(q_index: int, coordinate_name: str) -> dict[str, Any]:
    if q_index == 0:
        axis = "world +X"
        return {
            "q_meaning": "Floating-base position x in world frame.",
            "dq_meaning": "Floating-base COM linear velocity x in world frame.",
            "ddq_meaning": "Adjacent-step floating-base COM linear acceleration x in world frame.",
            "tau_meaning": (
                "World-frame actuation generalized force x component in N. This should be zero unless the "
                "base is directly actuated."
            ),
            "force_frame": "world",
            "positive_direction": axis,
            "sign_convention": "Isaac root world position and world-force convention",
        }
    if q_index == 1:
        axis = "world +Y"
        return {
            "q_meaning": "Floating-base position y in world frame.",
            "dq_meaning": "Floating-base COM linear velocity y in world frame.",
            "ddq_meaning": "Adjacent-step floating-base COM linear acceleration y in world frame.",
            "tau_meaning": (
                "World-frame actuation generalized force y component in N. This should be zero unless the "
                "base is directly actuated."
            ),
            "force_frame": "world",
            "positive_direction": axis,
            "sign_convention": "Isaac root world position and world-force convention",
        }
    if q_index == 2:
        axis = "world +Z"
        return {
            "q_meaning": "Floating-base position z in world frame.",
            "dq_meaning": "Floating-base COM linear velocity z in world frame.",
            "ddq_meaning": "Adjacent-step floating-base COM linear acceleration z in world frame.",
            "tau_meaning": (
                "World-frame actuation generalized force z component in N. This should be zero unless the "
                "base is directly actuated."
            ),
            "force_frame": "world",
            "positive_direction": axis,
            "sign_convention": "Isaac root world position and world-force convention",
        }
    angular_axis_by_index = {3: "world +X", 4: "world +Y", 5: "world +Z"}
    return {
        "q_meaning": f"Floating-base {coordinate_name} Euler angle from the Isaac root quaternion.",
        "dq_meaning": (
            "Floating-base angular velocity vector component in world frame. This is not an Euler-angle rate."
        ),
        "ddq_meaning": (
            "Adjacent-step floating-base angular acceleration vector component in world frame. "
            "This is not an Euler-angle second derivative."
        ),
        "tau_meaning": (
            "World-frame actuation generalized torque component in N*m, positive about the listed world axis. "
            "This should be zero unless the base is directly actuated."
        ),
        "force_frame": "world",
        "positive_direction": angular_axis_by_index.get(q_index, "world angular axis"),
        "sign_convention": (
            "q stores roll/pitch/yaw for Identix compatibility; dq, ddq, and tau use world angular-vector axes"
        ),
    }


def _database_contract(
    *,
    metadata: dict[str, Any],
    sim_table_name: str,
    sim_columns: list[str],
    dynamics_columns: list[str],
    coordinate_contract: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "intended_consumer": "Identix/system-identification tooling",
        "row_alignment": (
            "sysid sim_data rowid - 1 equals dynamics_data.sample_id. The optional source debug DB keeps the "
            "original unfiltered sample_id."
        ),
        "selected_model": SYSID_MODEL,
        "plain_language_summary": (
            "Each row stores the full floating-base robot state, both legs, and an actuation-only "
            "generalized-force label. The first six coordinates are the floating base; the remaining "
            "coordinates are the listed Isaac joints. The tau columns in sim_data are actuation only, by "
            "Identix convention. The full measured external side of the physics balance is kept separately in "
            "dynamics_data tau_external. The ddq columns use "
            f"{metadata.get('ddq_source', 'the source recording acceleration convention')}."
        ),
        "sim_data": {
            "table_name": sim_table_name,
            "columns": sim_columns,
            "field_blocks": {
                "q": {
                    "columns": [f"q{index}" for index in range(len(coordinate_contract))],
                    "meaning": "generalized position coordinates in coordinate_contract order",
                },
                "dq": {
                    "columns": [f"dq{index}" for index in range(len(coordinate_contract))],
                    "meaning": "generalized velocity coordinates in coordinate_contract order",
                },
                "ddq": {
                    "columns": [f"ddq{index}" for index in range(len(coordinate_contract))],
                    "meaning": (
                        "generalized acceleration coordinates in coordinate_contract order, using metadata ddq_source"
                    ),
                },
                "tau": {
                    "columns": [f"tau{index}" for index in range(len(coordinate_contract))],
                    "meaning": (
                        "tau_actuation only. Contact and friction are intentionally not included in sim_data tau "
                        "because Identix accounts for them through separate model inputs/conventions."
                    ),
                },
            },
        },
        "dynamics_data": {
            "table_name": "dynamics_data",
            "columns": dynamics_columns,
            "terms": {
                "tau_inertia": "M(q) * ddq using the same acceleration source stored in sim_data.",
                "tau_coriolis": "Coriolis/centrifugal generalized force from the PhysX force API.",
                "tau_gravity": (
                    "PhysX gravity generalized force corrected by subtracting the base-only compensation-derived "
                    "gravity wrench."
                ),
                "tau_tendon": "Analytic/projected tendon generalized force on the model side.",
                "tau_actuation": "Controller/motor generalized actuation command used as an external force.",
                "tau_contact": (
                    "Measured contact_validated generalized force from sensor force plus accepted contact-point "
                    "moment. The moment convention is force x (contact_pos_w - body_pos_w), matching the contact "
                    "sensor force sign to the angular Jacobian wrench convention."
                ),
                "tau_friction": "Configured joint-friction generalized force; zero in the final run.",
                "tau_external": (
                    "tau_actuation + tau_contact + tau_friction; used by the residual but intentionally separate "
                    "from sim_data tau."
                ),
            },
            "residual": {
                "formula": ("residual = tau_inertia + tau_gravity + tau_coriolis + tau_tendon - tau_external"),
                "norm": "Euclidean norm over all generalized coordinates.",
                "sign": "positive coordinate residual means model/conservative side exceeds measured external side.",
            },
        },
        "coordinate_contract": coordinate_contract,
        "force_orientation": {
            "base_linear": (
                "sim_data tau0, tau1, tau2 are world-frame actuation force components along +X, +Y, +Z; "
                "these are expected to be zero for an unactuated floating base."
            ),
            "base_angular": (
                "sim_data tau3, tau4, tau5 are world-frame actuation torque components about +X, +Y, +Z; "
                "these are expected to be zero for an unactuated floating base."
            ),
            "joints": (
                "sim_data tau6..tauN are actuation generalized torques in the positive Isaac "
                "joint-coordinate direction."
            ),
        },
        "source_files": {
            "source_metadata": metadata.get("metadata_path", "metadata.json"),
            "source_debug_db": metadata.get("debug_sqlite_path", "debug.db"),
        },
    }


def _kinematic_quality_by_sample_id(
    sim_rows: list[tuple[int, tuple[float, ...]]],
    debug_rows: list[dict[str, Any]],
    num_dofs: int,
    metadata: dict[str, Any],
) -> dict[int, dict[str, float]]:
    sim_dt = float(metadata.get("sim_dt", 0.0))
    expected_step_delta = int(metadata.get("config", {}).get("sampling_stride", 1))
    if sim_dt <= 0.0 or expected_step_delta <= 0:
        return {}

    mappings = sorted(metadata.get("coordinate_mappings", []), key=lambda item: int(item.get("q_index", -1)))
    skip_q_indices = {
        int(mapping["q_index"])
        for mapping in mappings
        if mapping.get("coordinate_type") == "floating_base"
        and str(mapping.get("coordinate_name")) in {"base_roll", "base_pitch", "base_yaw"}
    }
    angular_q_indices = {
        int(mapping["q_index"])
        for mapping in mappings
        if int(mapping["q_index"]) not in skip_q_indices and mapping.get("q_unit", mapping.get("units")) == "rad"
    }
    sim_by_sample_id = dict(sim_rows)
    samples_by_stream: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for row in debug_rows:
        sample_id = int(row["sample_id"])
        if sample_id not in sim_by_sample_id:
            continue
        key = (int(row["env_id"]), str(row["side"]))
        samples_by_stream.setdefault(key, []).append((int(row["step_index"]), sample_id))

    quality: dict[int, dict[str, float]] = {}
    for samples in samples_by_stream.values():
        samples.sort()
        for index in range(1, len(samples)):
            previous_step, previous_sample_id = samples[index - 1]
            step, sample_id = samples[index]
            step_delta = step - previous_step
            if step_delta != expected_step_delta:
                continue
            dt = float(step_delta) * sim_dt
            previous_values = sim_by_sample_id[previous_sample_id]
            values = sim_by_sample_id[sample_id]
            q_step_error = []
            ddq_backward_error = []
            for dof_index in range(num_dofs):
                if dof_index not in skip_q_indices:
                    q_delta = values[dof_index] - previous_values[dof_index]
                    if dof_index in angular_q_indices:
                        q_delta = _wrap_to_pi(q_delta)
                    expected_delta = 0.5 * (previous_values[num_dofs + dof_index] + values[num_dofs + dof_index]) * dt
                    q_step_error.append(q_delta - expected_delta)
                dq_delta = values[num_dofs + dof_index] - previous_values[num_dofs + dof_index]
                ddq_backward_error.append(values[2 * num_dofs + dof_index] - dq_delta / dt)
            q_step_norm = _vector_norm(tuple(q_step_error))
            ddq_backward_norm = _vector_norm(tuple(ddq_backward_error))
            quality[sample_id] = {
                "q_step_norm": q_step_norm,
                "ddq_backward_norm": ddq_backward_norm,
                "quality_norm": q_step_norm,
                "combined_kinematic_norm": max(q_step_norm, ddq_backward_norm),
            }
    return quality


def _residual_cleanliness_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    residuals = [float(record["residual_norm"]) for record in records]
    if not residuals:
        return {}
    ordered = sorted(residuals)
    p90 = _percentile_sorted(ordered, 90.0)
    p95 = _percentile_sorted(ordered, 95.0)
    p99 = _percentile_sorted(ordered, 99.0)
    high_records = [record for record in records if float(record["residual_norm"]) >= p95]
    quality_records = [record for record in records if "quality_norm" in record]
    high_quality_records = [record for record in high_records if "quality_norm" in record]
    quality_values = [float(record["quality_norm"]) for record in quality_records]
    quality_ordered = sorted(quality_values)
    quality_p95 = _percentile_sorted(quality_ordered, 95.0) if quality_ordered else 0.0
    quality_p99 = _percentile_sorted(quality_ordered, 99.0) if quality_ordered else 0.0
    high_above_quality_p95 = [record for record in high_quality_records if float(record["quality_norm"]) >= quality_p95]
    high_above_quality_p99 = [record for record in high_quality_records if float(record["quality_norm"]) >= quality_p99]
    high_fraction_above_p95 = len(high_above_quality_p95) / max(len(high_quality_records), 1)

    return {
        "definition": (
            "cleanliness is computed before any residual filter. High residual rows are the top 5% by final "
            "sysid residual norm."
        ),
        "row_count": len(records),
        "residual_norm": _norm_summary(residuals),
        "thresholds": [
            _threshold_count(records, p90, "p90"),
            _threshold_count(records, p95, "p95"),
            _threshold_count(records, p99, "p99"),
            _threshold_count(records, 10.0, "absolute_10"),
            _threshold_count(records, 100.0, "absolute_100"),
            _threshold_count(records, 300.0, "absolute_300"),
            _threshold_count(records, 500.0, "absolute_500"),
            _threshold_count(records, 750.0, "absolute_750"),
            _threshold_count(records, 1000.0, "absolute_1000"),
            _threshold_count(records, 1200.0, "absolute_1200"),
        ],
        "high_residual_rows": {
            "threshold": p95,
            "count": len(high_records),
            "fraction": len(high_records) / max(len(records), 1),
            "unique_env_count": len({int(record["env_id"]) for record in high_records}),
            "unique_step_count": len({int(record["step_index"]) for record in high_records}),
            "top_envs": _top_counts(high_records, "env_id", limit=10),
            "top_steps": _top_counts(high_records, "step_index", limit=10),
            "longest_consecutive_run_by_env": _longest_consecutive_high_run(high_records),
        },
        "kinematic_quality": {
            "available_rows": len(quality_records),
            "missing_rows": len(records) - len(quality_records),
            "quality_norm_definition": "q_step_norm; ddq finite-difference disagreement is diagnostic only",
            "overall_quality_norm": _norm_summary(quality_values),
            "high_residual_quality_norm": _norm_summary(
                [float(record["quality_norm"]) for record in high_quality_records]
            ),
            "overall_p95": quality_p95,
            "overall_p99": quality_p99,
            "high_residual_rows_above_overall_p95": len(high_above_quality_p95),
            "high_residual_fraction_above_overall_p95": high_fraction_above_p95,
            "high_residual_rows_above_overall_p99": len(high_above_quality_p99),
            "high_residual_fraction_above_overall_p99": len(high_above_quality_p99) / max(len(high_quality_records), 1),
            "interpretation": _kinematic_breakdown_interpretation(high_fraction_above_p95),
        },
    }


def _threshold_count(records: list[dict[str, Any]], threshold: float, label: str) -> dict[str, Any]:
    count = sum(1 for record in records if float(record["residual_norm"]) > threshold)
    fraction = count / max(len(records), 1)
    below_or_equal = len(records) - count
    return {
        "label": label,
        "threshold": threshold,
        "comparison": ">",
        "rows_above": count,
        "fraction_above": fraction,
        "rows_at_or_above": count,
        "fraction_at_or_above": fraction,
        "rows_below_or_equal": below_or_equal,
        "fraction_below_or_equal": below_or_equal / max(len(records), 1),
    }


def _top_counts(records: list[dict[str, Any]], key: str, *, limit: int) -> list[dict[str, int]]:
    counts: dict[int, int] = {}
    for record in records:
        value = int(record[key])
        counts[value] = counts.get(value, 0) + 1
    return [
        {key: value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _longest_consecutive_high_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    steps_by_env: dict[int, list[int]] = {}
    for record in records:
        steps_by_env.setdefault(int(record["env_id"]), []).append(int(record["step_index"]))
    best_env = None
    best_run = 0
    for env_id, steps in steps_by_env.items():
        unique_steps = sorted(set(steps))
        current_run = 0
        previous_step = None
        for step in unique_steps:
            current_run = current_run + 1 if previous_step is not None and step == previous_step + 1 else 1
            previous_step = step
            if current_run > best_run:
                best_run = current_run
                best_env = env_id
    return {"env_id": best_env, "length_steps": best_run}


def _kinematic_breakdown_interpretation(high_fraction_above_p95: float) -> str:
    if high_fraction_above_p95 >= 0.5:
        return "high residual rows are strongly associated with poor adjacent-step kinematic consistency"
    if high_fraction_above_p95 >= 0.2:
        return "high residual rows are partially associated with kinematic inconsistency"
    return "high residual rows are not mostly explained by adjacent-step kinematic breakdown"


def _sysid_terms(debug_row: dict[str, Any], num_dofs: int, base_indices: list[int]) -> dict[str, tuple[float, ...]]:
    _ = base_indices
    return {
        "inertia": _term(debug_row, "inertia", num_dofs),
        "coriolis": _term(debug_row, "coriolis", num_dofs),
        "gravity": _term(debug_row, "gravity_identification", num_dofs),
        "tendon": _term(debug_row, "tendon", num_dofs),
        "actuation": _term(debug_row, "actuation_command", num_dofs),
        "contact": _term(debug_row, "contact_validated", num_dofs),
        "friction": _term(debug_row, "friction", num_dofs),
        "solver_constraint_internal": _term(debug_row, "solver_constraint_internal", num_dofs),
    }


def _term(row: dict[str, Any], name: str, num_dofs: int) -> tuple[float, ...]:
    return tuple(float(row[f"tau_{name}{index}"]) for index in range(num_dofs))


def _read_sim_rows(path: Path, table_name: str, columns: list[str]) -> list[tuple[int, tuple[float, ...]]]:
    with sqlite3.connect(path) as db:
        actual_columns = _table_columns(db, table_name)
        if actual_columns != columns:
            raise ExportError(f"Unexpected sim_data columns in {path}")
        select_columns = ", ".join(_quote_identifier(name) for name in columns)
        rows = db.execute(
            f"SELECT rowid - 1, {select_columns} FROM {_quote_identifier(table_name)} ORDER BY rowid"
        ).fetchall()
    return [(int(row[0]), tuple(float(value) for value in row[1:])) for row in rows]


def _read_debug_rows(path: Path, num_dofs: int) -> tuple[list[dict[str, Any]], list[str]]:
    required_terms = (
        "inertia",
        "coriolis",
        "gravity_identification",
        "tendon",
        "actuation_command",
        "contact_validated",
        "friction",
    )
    selected_columns = ["sample_id", "step_index", "time", "env_id", "side"] + [
        f"tau_{term}{index}" for term in required_terms for index in range(num_dofs)
    ]
    with sqlite3.connect(path) as db:
        actual_columns = _table_columns(db, "debug_data")
        missing = [column for column in selected_columns if column not in actual_columns]
        if missing:
            raise ExportError(f"debug_data is missing required columns: {missing}")
        select_columns = ", ".join(_quote_identifier(name) for name in selected_columns)
        rows = db.execute(f"SELECT {select_columns} FROM debug_data ORDER BY sample_id").fetchall()
    return [dict(zip(selected_columns, row, strict=True)) for row in rows], selected_columns


def _write_kinematics_db(path: Path, table_name: str, columns: list[str], rows: list[tuple[float, ...]]) -> None:
    with sqlite3.connect(path) as db:
        columns_sql = ", ".join(f"{_quote_identifier(name)} REAL NOT NULL" for name in columns)
        db.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({columns_sql})")
        placeholders = ", ".join("?" for _ in columns)
        insert_columns = ", ".join(_quote_identifier(name) for name in columns)
        db.executemany(
            f"INSERT INTO {_quote_identifier(table_name)} ({insert_columns}) VALUES ({placeholders})",
            rows,
        )
        db.commit()


def _write_dynamics_db(path: Path, columns: list[str], rows: list[tuple[Any, ...]]) -> None:
    with sqlite3.connect(path) as db:
        columns_sql = ", ".join(_dynamics_column_sql(name) for name in columns)
        db.execute(f"CREATE TABLE dynamics_data ({columns_sql})")
        db.execute("CREATE INDEX dynamics_data_step_idx ON dynamics_data (step_index, side)")
        placeholders = ", ".join("?" for _ in columns)
        insert_columns = ", ".join(_quote_identifier(name) for name in columns)
        db.executemany(f"INSERT INTO dynamics_data ({insert_columns}) VALUES ({placeholders})", rows)
        db.commit()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExportError(f"Metadata file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _recording_file(recording_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else recording_dir / path.name


def _floating_base_indices(metadata: dict[str, Any]) -> list[int]:
    indices = [
        int(mapping["q_index"])
        for mapping in metadata.get("coordinate_mappings", [])
        if mapping.get("coordinate_type") == "floating_base"
    ]
    if not indices:
        raise ExportError("Source metadata has no floating-base coordinate mappings.")
    return sorted(indices)


def _table_columns(db: sqlite3.Connection, table_name: str) -> list[str]:
    rows = db.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    if not rows:
        raise ExportError(f"Missing table: {table_name}")
    return [str(row[1]) for row in rows]


def _sim_data_columns(num_dofs: int) -> list[str]:
    return (
        [f"q{i}" for i in range(num_dofs)]
        + [f"dq{i}" for i in range(num_dofs)]
        + [f"ddq{i}" for i in range(num_dofs)]
        + [f"tau{i}" for i in range(num_dofs)]
    )


def _dynamics_columns(num_dofs: int) -> list[str]:
    return ["sample_id", "step_index", "time", "env_id", "side"] + [
        f"tau_{name}{index}" for name in DYNAMICS_TERMS for index in range(num_dofs)
    ]


def _dynamics_column_sql(name: str) -> str:
    if name == "sample_id":
        return f"{_quote_identifier(name)} INTEGER PRIMARY KEY"
    if name in {"step_index", "env_id"}:
        return f"{_quote_identifier(name)} INTEGER NOT NULL"
    if name == "side":
        return f"{_quote_identifier(name)} TEXT NOT NULL"
    return f"{_quote_identifier(name)} REAL NOT NULL"


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _vector_add(*vectors: tuple[float, ...]) -> tuple[float, ...]:
    width = len(vectors[0])
    return tuple(sum(vector[index] for vector in vectors) for index in range(width))


def _vector_sub(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(left[index] - right[index] for index in range(len(left)))


def _vector_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _norm_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": _percentile_sorted(ordered, 50.0),
        "p95": _percentile_sorted(ordered, 95.0),
        "max": max(values),
    }


def _percentile_sorted(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    fraction = rank - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


if __name__ == "__main__":
    raise SystemExit(main())
