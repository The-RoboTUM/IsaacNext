# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual candidate equations for generic dynamics audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WeightedTerm = tuple[str, float]


@dataclass(frozen=True)
class DynamicsCandidate:
    """One residual equation candidate."""

    label: str
    family: str
    lhs_terms: tuple[WeightedTerm, ...]
    rhs_terms: tuple[WeightedTerm, ...]
    role: str = "diagnostic_control"
    note: str = ""


def default_dynamics_candidates() -> tuple[DynamicsCandidate, ...]:
    """Return the generic candidate set used by small-system audits."""

    selected_lhs = (
        ("inertia", 1.0),
        ("gravity_compensation_actual", -1.0),
        ("coriolis_compensation_actual", -1.0),
    )
    direct_lhs = (
        ("inertia", 1.0),
        ("gravity_force_api", 1.0),
        ("coriolis_force_api", 1.0),
    )
    positive_compensation_lhs = (
        ("inertia", 1.0),
        ("gravity_compensation_actual", 1.0),
        ("coriolis_compensation_actual", 1.0),
    )
    selected_rhs = (("actuation_command", 1.0),)
    candidates = [
        DynamicsCandidate(
            "selected_negative_compensation",
            "baseline",
            selected_lhs,
            selected_rhs,
            role="model_candidate",
            note="Mddq - gravity_compensation_actual - coriolis_compensation_actual = command effort",
        ),
        DynamicsCandidate(
            "direct_force_api",
            "force_source",
            direct_lhs,
            selected_rhs,
            role="model_candidate",
            note="Mddq + direct PhysX gravity + direct PhysX Coriolis = command effort",
        ),
        DynamicsCandidate(
            "positive_compensation_actual",
            "force_source",
            positive_compensation_lhs,
            selected_rhs,
            note="positive-sign compensation alias check; this was bad for Forrest",
        ),
        DynamicsCandidate(
            "direct_gravity_negative_coriolis_comp",
            "force_source",
            (
                ("inertia", 1.0),
                ("gravity_force_api", 1.0),
                ("coriolis_compensation_actual", -1.0),
            ),
            selected_rhs,
        ),
        DynamicsCandidate(
            "negative_gravity_comp_direct_coriolis",
            "force_source",
            (
                ("inertia", 1.0),
                ("gravity_compensation_actual", -1.0),
                ("coriolis_force_api", 1.0),
            ),
            selected_rhs,
        ),
        DynamicsCandidate(
            "negative_gravity_comp_only",
            "term_removal",
            (("inertia", 1.0), ("gravity_compensation_actual", -1.0)),
            selected_rhs,
        ),
        DynamicsCandidate(
            "negative_coriolis_comp_only",
            "term_removal",
            (("inertia", 1.0), ("coriolis_compensation_actual", -1.0)),
            selected_rhs,
        ),
        DynamicsCandidate("inertia_only", "term_removal", (("inertia", 1.0),), selected_rhs),
        DynamicsCandidate(
            "flip_direct_gravity",
            "sign_check",
            (
                ("inertia", 1.0),
                ("gravity_force_api", -1.0),
                ("coriolis_force_api", 1.0),
            ),
            selected_rhs,
        ),
        DynamicsCandidate(
            "flip_direct_coriolis",
            "sign_check",
            (
                ("inertia", 1.0),
                ("gravity_force_api", 1.0),
                ("coriolis_force_api", -1.0),
            ),
            selected_rhs,
        ),
        DynamicsCandidate(
            "selected_with_applied_torque",
            "actuation_source",
            selected_lhs,
            (("applied_torque", 1.0),),
            role="model_candidate",
            note="uses IsaacLab robot.data.applied_torque as the actuation source",
        ),
        DynamicsCandidate(
            "selected_with_computed_torque",
            "actuation_source",
            selected_lhs,
            (("computed_torque", 1.0),),
            note="uses IsaacLab robot.data.computed_torque as the actuation source",
        ),
        DynamicsCandidate(
            "selected_with_joint_effort_target",
            "actuation_source",
            selected_lhs,
            (("joint_effort_target", 1.0),),
        ),
        DynamicsCandidate(
            "selected_with_physx_actuation",
            "actuation_source",
            selected_lhs,
            (("physx_actuation", 1.0),),
            note="uses root_physx_view.get_dof_actuation_forces when exposed",
        ),
        DynamicsCandidate(
            "selected_with_implicit_drive_estimate",
            "actuation_source",
            selected_lhs,
            (("implicit_drive_estimate", 1.0),),
            note="uses stiffness/damping/effort target reconstruction of implicit drives",
        ),
        DynamicsCandidate(
            "selected_with_previous_command",
            "timing",
            selected_lhs,
            (("actuation_previous_command", 1.0),),
            note="one-step command timing diagnostic",
        ),
        DynamicsCandidate(
            "selected_zero_actuation",
            "actuation_source",
            selected_lhs,
            (),
            note="diagnostic for free response / missing actuation",
        ),
        DynamicsCandidate(
            "selected_plus_solver_rhs",
            "solver",
            selected_lhs,
            (*selected_rhs, ("solver_joint", 1.0)),
            note="adds all projected joint solver forces to the RHS",
        ),
        DynamicsCandidate(
            "selected_minus_solver_rhs",
            "solver",
            selected_lhs,
            (*selected_rhs, ("solver_joint", -1.0)),
            note="opposite-sign all projected joint solver force check",
        ),
        DynamicsCandidate(
            "selected_plus_solver_lhs",
            "solver",
            (*selected_lhs, ("solver_joint", 1.0)),
            selected_rhs,
            note="adds all projected joint solver forces to the LHS",
        ),
        DynamicsCandidate(
            "selected_minus_solver_lhs",
            "solver",
            (*selected_lhs, ("solver_joint", -1.0)),
            selected_rhs,
            note="opposite-sign projected joint solver force on the LHS",
        ),
        DynamicsCandidate(
            "direct_plus_solver_rhs",
            "solver",
            direct_lhs,
            (*selected_rhs, ("solver_joint", 1.0)),
        ),
        DynamicsCandidate(
            "direct_minus_solver_rhs",
            "solver",
            direct_lhs,
            (*selected_rhs, ("solver_joint", -1.0)),
        ),
        DynamicsCandidate(
            "fd_acc_selected_negative_compensation",
            "acceleration_source",
            (
                ("inertia_fd", 1.0),
                ("gravity_compensation_actual", -1.0),
                ("coriolis_compensation_actual", -1.0),
            ),
            selected_rhs,
            note="uses finite-difference acceleration from recorded joint velocities",
        ),
        DynamicsCandidate(
            "fd_acc_direct_force_api",
            "acceleration_source",
            (
                ("inertia_fd", 1.0),
                ("gravity_force_api", 1.0),
                ("coriolis_force_api", 1.0),
            ),
            selected_rhs,
        ),
    ]
    return tuple(candidates)


def summarize_residual_candidates(
    rows: list[dict[str, tuple[float, ...]]],
    *,
    candidates: tuple[DynamicsCandidate, ...] | None = None,
    coordinate_names: tuple[str, ...] = (),
    groups: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Rank residual candidates over recorded term rows."""

    candidates = default_dynamics_candidates() if candidates is None else candidates
    groups = groups or {}
    active = (
        [
            candidate
            for candidate in candidates
            if all(_term_name(term) in rows[0] for term in (*candidate.lhs_terms, *candidate.rhs_terms))
        ]
        if rows
        else []
    )
    candidate_norms: dict[str, list[float]] = {candidate.label: [] for candidate in active}
    candidate_group_norms: dict[str, dict[str, list[float]]] = {
        candidate.label: {group: [] for group in groups} for candidate in active
    }
    candidate_worst: dict[str, tuple[float, int, tuple[float, ...]]] = {}

    for row_index, row in enumerate(rows):
        width = len(next(iter(row.values()))) if row else 0
        for candidate in active:
            residual = _vector_sub(
                _weighted_vector_sum(row, candidate.lhs_terms, width),
                _weighted_vector_sum(row, candidate.rhs_terms, width),
            )
            norm = _vector_norm(residual)
            candidate_norms[candidate.label].append(norm)
            for group_name, indices in groups.items():
                candidate_group_norms[candidate.label][group_name].append(_vector_norm(residual, indices))
            if candidate.label not in candidate_worst or norm > candidate_worst[candidate.label][0]:
                candidate_worst[candidate.label] = (norm, row_index, residual)

    summaries = []
    meta = {candidate.label: candidate for candidate in active}
    for label, norms in candidate_norms.items():
        if not norms:
            continue
        candidate = meta[label]
        summaries.append(
            {
                "label": label,
                "family": candidate.family,
                "role": candidate.role,
                "note": candidate.note,
                **_norm_summary(norms),
                "group_mean_norms": {
                    group: _mean(values) for group, values in candidate_group_norms[label].items() if values
                },
            }
        )
    summaries.sort(key=lambda item: (item["mean"], item["p95"], item["max"]))
    best = summaries[0] if summaries else None
    worst = None
    if best is not None:
        _, row_index, residual = candidate_worst[best["label"]]
        worst = {
            "candidate": best["label"],
            "row_index": row_index,
            "coordinates": [
                {
                    "index": index,
                    "name": coordinate_names[index] if index < len(coordinate_names) else f"q{index}",
                    "residual": value,
                }
                for index, value in enumerate(residual)
            ],
        }
    return {
        "rows": len(rows),
        "candidate_summaries": summaries,
        "best_candidate": best,
        "worst_row_for_best": worst,
        "term_diagnostics": _term_diagnostics(rows),
    }


def print_residual_report(summary: dict[str, Any], *, limit: int = 20) -> None:
    """Print a compact residual ranking."""

    print("\n[SimpleDynamicsAudit] Residual experiments")
    print("  residual form: conservative_terms - non_conservative_terms")
    print(f"  rows: {summary.get('rows', 0)}")
    print("  ranked candidates by mean residual, lower is better:")
    for rank, item in enumerate(summary.get("candidate_summaries", [])[:limit], start=1):
        suffix = _format_group_suffix(item.get("group_mean_norms", {}))
        print(
            f"    {rank:2d}. {item['label']:<44} "
            f"family={item['family']:<20} mean={item['mean']:10.6f} "
            f"p95={item['p95']:10.6f} max={item['max']:10.6f}{suffix}"
        )
    diagnostics = summary.get("term_diagnostics", {})
    if diagnostics:
        print("  diagnostic deltas and term magnitudes:")
        for name, item in diagnostics.items():
            print(f"    {name:<45} mean={item['mean']:10.6f} p95={item['p95']:10.6f} max={item['max']:10.6f}")
    worst = summary.get("worst_row_for_best")
    if worst:
        print(f"  worst row for best candidate {worst['candidate']!r}: row_index={worst['row_index']}")
        for coord in worst["coordinates"]:
            print(f"    q{coord['index']:<2d} {coord['name']:<32} {coord['residual']:+12.6f}")


def _term_diagnostics(rows: list[dict[str, tuple[float, ...]]]) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, list[float]] = {
        "inertia_minus_inertia_fd": [],
        "gravity_force_api_minus_negative_compensation": [],
        "coriolis_force_api_minus_negative_compensation": [],
        "command_minus_applied_torque": [],
        "command_minus_computed_torque": [],
        "command_minus_physx_actuation": [],
        "command_minus_implicit_drive": [],
        "solver_joint_norm": [],
        "implicit_drive_saturation_norm": [],
    }
    for row in rows:
        _append_delta(diagnostics, row, "inertia_minus_inertia_fd", "inertia", "inertia_fd")
        if "gravity_force_api" in row and "gravity_compensation_actual" in row:
            diagnostics["gravity_force_api_minus_negative_compensation"].append(
                _vector_norm(_vector_add(row["gravity_force_api"], row["gravity_compensation_actual"]))
            )
        if "coriolis_force_api" in row and "coriolis_compensation_actual" in row:
            diagnostics["coriolis_force_api_minus_negative_compensation"].append(
                _vector_norm(_vector_add(row["coriolis_force_api"], row["coriolis_compensation_actual"]))
            )
        _append_delta(diagnostics, row, "command_minus_applied_torque", "actuation_command", "applied_torque")
        _append_delta(diagnostics, row, "command_minus_computed_torque", "actuation_command", "computed_torque")
        _append_delta(diagnostics, row, "command_minus_physx_actuation", "actuation_command", "physx_actuation")
        _append_delta(diagnostics, row, "command_minus_implicit_drive", "actuation_command", "implicit_drive_estimate")
        if "solver_joint" in row:
            diagnostics["solver_joint_norm"].append(_vector_norm(row["solver_joint"]))
        if "implicit_drive_saturation" in row:
            diagnostics["implicit_drive_saturation_norm"].append(_vector_norm(row["implicit_drive_saturation"]))
    return {name: _norm_summary(values) for name, values in diagnostics.items() if values}


def _append_delta(
    diagnostics: dict[str, list[float]],
    row: dict[str, tuple[float, ...]],
    label: str,
    left: str,
    right: str,
) -> None:
    if left in row and right in row:
        diagnostics[label].append(_vector_norm(_vector_sub(row[left], row[right])))


def _weighted_vector_sum(
    row: dict[str, tuple[float, ...]],
    weighted_terms: tuple[WeightedTerm, ...],
    width: int,
) -> tuple[float, ...]:
    result = [0.0] * width
    for term_name, scale in weighted_terms:
        values = row[term_name]
        for index, value in enumerate(values):
            result[index] += float(scale) * float(value)
    return tuple(result)


def _term_name(term: WeightedTerm) -> str:
    return str(term[0])


def _vector_add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(left[index]) + float(right[index]) for index in range(len(left)))


def _vector_sub(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(left[index]) - float(right[index]) for index in range(len(left)))


def _vector_norm(values: tuple[float, ...], indices: list[int] | None = None) -> float:
    selected = values if indices is None else tuple(values[index] for index in indices)
    if not selected:
        return 0.0
    return float(sum(float(value) * float(value) for value in selected) ** 0.5)


def _norm_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(float(value) for value in values)
    p95_index = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return {
        "mean": _mean(sorted_values),
        "p95": sorted_values[p95_index],
        "max": sorted_values[-1],
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _format_group_suffix(group_mean_norms: dict[str, float]) -> str:
    selected = []
    for group_name in ("all", "cart", "pole", "pendulum"):
        if group_name in group_mean_norms:
            selected.append(f"{group_name}={group_mean_norms[group_name]:.6f}")
    return "" if not selected else " | " + ", ".join(selected)
