# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Patch Forrest USD layers after URDF import.

The current Forrest URDF import can leave stale references from the base layer to
virtual anchor visual prims that are missing in the physics layer. When cloned
across many RL environments, USD reports the missing references once per clone.
This script adds empty Xform prims at the referenced paths so composition stays
clean without suppressing unrelated USD warnings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf

DEFAULT_PHYSICS_LAYER = (
    Path("symlinks") / "forrest_ws" / "urdf" / "forrest_isaac" / "configuration" / "forrest_isaac_physics.usd"
)

REQUIRED_VIRTUAL_VISUALS = (
    "s23_assy_1_virtual",
    "s23_assy_2_virtual",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Forrest USD virtual visual reference targets.")
    parser.add_argument(
        "--physics-layer",
        type=Path,
        default=DEFAULT_PHYSICS_LAYER,
        help=f"Path to forrest_isaac_physics.usd. Default: {DEFAULT_PHYSICS_LAYER}",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether the patch is present. Do not modify the USD file.",
    )
    args = parser.parse_args()

    physics_layer_path = args.physics_layer.expanduser()
    if not physics_layer_path.exists():
        raise FileNotFoundError(
            f"Forrest physics USD layer not found: {physics_layer_path}. "
            "Run scripts/setup_repo.sh and convert the Forrest URDF to USD first."
        )

    layer = Sdf.Layer.FindOrOpen(str(physics_layer_path))
    if layer is None:
        raise RuntimeError(f"Could not open USD layer: {physics_layer_path}")

    missing_paths = _missing_virtual_visual_paths(layer)
    if args.check:
        if missing_paths:
            raise RuntimeError(
                "Forrest USD patch is missing these prims: " + ", ".join(str(path) for path in missing_paths)
            )
        print(f"Forrest USD patch already present: {physics_layer_path}")
        return

    if not missing_paths:
        print(f"Forrest USD patch already present: {physics_layer_path}")
        return

    Sdf.CreatePrimInLayer(layer, "/visuals")
    for prim_path in missing_paths:
        prim = Sdf.CreatePrimInLayer(layer, str(prim_path))
        prim.specifier = Sdf.SpecifierDef
        prim.typeName = "Xform"

    if not layer.Save():
        raise RuntimeError(f"Could not save USD layer: {physics_layer_path}")

    print("Patched Forrest USD virtual visuals:")
    for prim_path in missing_paths:
        print(f"  {prim_path}")
    print(f"Saved: {physics_layer_path}")


def _missing_virtual_visual_paths(layer: Sdf.Layer) -> list[Sdf.Path]:
    missing_paths = []
    for name in REQUIRED_VIRTUAL_VISUALS:
        prim_path = Sdf.Path(f"/visuals/{name}")
        if layer.GetPrimAtPath(prim_path) is None:
            missing_paths.append(prim_path)
    return missing_paths


if __name__ == "__main__":
    main()
