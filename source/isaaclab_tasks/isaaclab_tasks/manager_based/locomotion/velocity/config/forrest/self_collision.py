# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selective self-collision filtering for Forrest."""

from __future__ import annotations

from pxr import Sdf, Usd, UsdPhysics

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


@configclass
class SelectiveSelfCollisionCfg:
    """Filter all internal Forrest collisions except an allowed body set."""

    robot_path_template: str = "/World/envs/env_{env_id}/forrest_isaac"
    allowed_body_names: tuple[str, ...] = (
        "s23_assy_1",
        "s34_foot_connector_assy_1",
        "s45_digit_assy_1",
        "s23_assy_2",
        "s34_foot_connector_assy_2",
        "s45_digit_assy_2",
    )
    debug: bool = False


def create_selective_self_collision_filter(
    env: ManagerBasedRLEnv,
    env_ids,
    cfg: SelectiveSelfCollisionCfg | None = None,
) -> None:
    """Author USD filtered pairs so only selected Forrest body pairs self-collide.

    This function assumes articulation self-collisions are enabled. It then
    filters all internal rigid-body pairs except pairs where both body names are
    in ``allowed_body_names``.
    """
    del env_ids  # prestartup event; applies to all cloned environments.

    if cfg is None:
        cfg = SelectiveSelfCollisionCfg()

    stage = env.sim.get_initial_stage()
    allowed = set(cfg.allowed_body_names)
    filtered_pairs = 0
    allowed_pairs = 0

    for env_id in range(env.num_envs):
        robot_path = Sdf.Path(cfg.robot_path_template.format(env_id=env_id))
        robot_prim = stage.GetPrimAtPath(robot_path)
        if not robot_prim.IsValid():
            raise RuntimeError(f"Cannot filter Forrest self-collisions: robot prim does not exist: {robot_path}")

        bodies = _find_rigid_bodies(robot_prim)
        if not bodies:
            raise RuntimeError(f"Cannot filter Forrest self-collisions: no rigid bodies under {robot_path}")

        for i, body_a in enumerate(bodies):
            body_a_allowed = body_a.GetName() in allowed
            targets_a = _collision_filter_targets(body_a)

            for body_b in bodies[i + 1 :]:
                if body_a_allowed and body_b.GetName() in allowed:
                    allowed_pairs += 1
                    continue

                targets_b = _collision_filter_targets(body_b)
                for target_a in targets_a:
                    for target_b in targets_b:
                        _filter_pair(target_a, target_b)
                        _filter_pair(target_b, target_a)
                        filtered_pairs += 1

    if cfg.debug:
        print(
            "[ForrestSelfCollision] "
            f"allowed body names={tuple(sorted(allowed))}, "
            f"allowed body pairs={allowed_pairs}, filtered target pairs={filtered_pairs}."
        )


def _find_rigid_bodies(root_prim: Usd.Prim) -> list[Usd.Prim]:
    return [prim for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]


def _collision_filter_targets(body_prim: Usd.Prim) -> list[Usd.Prim]:
    targets = [prim for prim in Usd.PrimRange(body_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if targets:
        return targets
    return [body_prim]


def _filter_pair(prim_a: Usd.Prim, prim_b: Usd.Prim) -> None:
    rel = UsdPhysics.FilteredPairsAPI.Apply(prim_a).CreateFilteredPairsRel()
    rel.AddTarget(prim_b.GetPath())
