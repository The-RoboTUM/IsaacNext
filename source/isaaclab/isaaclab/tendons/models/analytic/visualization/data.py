# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Input/output helpers for tendon visualization."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from isaaclab.tendons.models.analytic.visualization.context import td, tids


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load frame-by-frame tendon debug data from a JSONL file."""
    path = Path(path)
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_recording(path: str | Path, *, side: str = "left") -> tuple[list[dict[str, Any]], str]:
    """Load either tendon debug frames or trajectory-only frames from a recording directory/DB."""

    path = Path(path)
    recording_dir = path if path.is_dir() else path.parent
    tendon_db = path if path.name == "forrest_tendons.db" else recording_dir / "forrest_tendons.db"
    kinematics_db = path if path.name == "forrest_kinematics.db" else recording_dir / "forrest_kinematics.db"

    if tendon_db.exists():
        frames = load_tendon_db(tendon_db, side=side)
        if frames:
            return frames, "tendon"
    if not kinematics_db.exists():
        raise FileNotFoundError(f"No forrest_tendons.db or forrest_kinematics.db found for {path}")
    return load_kinematics_db(kinematics_db, side=side), "trajectory"


def load_tendon_db(path: str | Path, *, side: str = "left") -> list[dict[str, Any]]:
    """Load JSON-compatible tendon debug frames from ``forrest_tendons.db``."""

    path = Path(path)
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT frame_json FROM tendon_frames WHERE side = ? ORDER BY step_index, time",
            (side,),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def load_kinematics_db(path: str | Path, *, side: str = "left") -> list[dict[str, Any]]:
    """Load trajectory-only frames from ``forrest_kinematics.db`` and metadata."""

    path = Path(path)
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    joint_names = _joint_names_for_side(metadata, side)
    q_indices = _tendon_chain_q_indices(joint_names, side)
    sim_dt = metadata.get("sim_dt") or 0.0

    with sqlite3.connect(path) as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(sim_data)").fetchall()]
        q_columns = [name for name in columns if name.startswith("q") and name[1:].isdigit()]
        q_columns.sort(key=lambda name: int(name[1:]))
        rows = db.execute(
            "SELECT " + ", ".join(f'"{name}"' for name in q_columns) + " FROM sim_data ORDER BY rowid"
        ).fetchall()

    frames = []
    for step_index, row in enumerate(rows, start=1):
        q_values = [float(value) for value in row]
        joint_angles = [q_values[index] for index in q_indices]
        frames.append(
            {
                "step_index": step_index,
                "sim_time": (step_index - 1) * float(sim_dt),
                "joint_pos": joint_angles,
                "thetas": _thetas_from_joint_angles(joint_angles),
            }
        )
    return frames


def _joint_names_for_side(metadata: dict[str, Any], side: str) -> list[str]:
    rows = [row for row in metadata["joint_mappings"] if row["side"] == side]
    rows.sort(key=lambda row: int(row["q_index"]))
    if not rows:
        raise ValueError(f"No joint mapping for side {side!r} in metadata.")
    return [row["joint_name"] for row in rows]


def _tendon_chain_q_indices(joint_names: list[str], side: str) -> list[int]:
    prefix = "l" if side == "left" else "r"
    required = [
        f"{prefix}3f_femorotibial_front",
        f"{prefix}4f_intertarsal_front",
        f"{prefix}5_metatarsophalangeal",
        f"{prefix}6_interphalangeal",
        f"{prefix}8_knee_flexor",
    ]
    missing = [name for name in required if name not in joint_names]
    if missing:
        raise ValueError(f"Kinematics DB is missing tendon-chain joints needed for playback: {missing}")
    return [joint_names.index(name) for name in required]


def _thetas_from_joint_angles(joint_angles: list[float]) -> list[float]:
    joint_ids = [
        tids.I_JOINT_3,
        tids.I_JOINT_4,
        tids.I_JOINT_5,
        tids.I_JOINT_6,
        tids.I_JOINT_5,
        tids.I_JOINT_4,
        tids.I_JOINT_5,
        tids.I_JOINT_4,
        tids.I_JOINT_5,
        tids.I_JOINT_3,
        tids.I_JOINT_8,
    ]
    values = [0.0] * int(td.tendon_offsets_theta.shape[1])
    for theta_id, joint_id in enumerate(joint_ids):
        joint_direction = (
            td.joint_directions[joint_id] if td.joint_directions.ndim == 1 else td.joint_directions[0, joint_id]
        )
        signed_angle = float(joint_direction.item()) * float(joint_angles[joint_id])
        values[theta_id] = signed_angle + float(td.tendon_offsets_theta[0, theta_id].item())
    return values
