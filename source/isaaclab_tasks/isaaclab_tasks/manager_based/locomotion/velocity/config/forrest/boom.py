# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planar boom constraint utilities for Forrest debug environments."""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass


@configclass
class BoomConstraintCfg:
    """USD D6-joint setup for a sagittal-plane boom.

    Forrest's current convention is front = -Y and up = +Z, so the sagittal
    plane is Y-Z. The boom locks lateral X translation and rotations about Y/Z.
    Set ``lock_x_angle`` to also lock rotation about X.
    """

    body_path_template: str = "/World/envs/env_{env_id}/forrest_urdf_latest/world_corrected"
    joint_path_template: str = "/World/envs/env_{env_id}/forrest_urdf_latest/world_corrected_planar_boom_joint"
    locked_axes: tuple[str, ...] = ("transX", "rotY", "rotZ")
    lock_x_angle: bool = False
    body_anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_anchor_rot_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    debug: bool = False


def _resolve_locked_axes(cfg: BoomConstraintCfg) -> tuple[str, ...]:
    if cfg.lock_x_angle and "rotX" not in cfg.locked_axes:
        return (*cfg.locked_axes, "rotX")
    return cfg.locked_axes


def _lock_d6_axis(joint_prim, axis: str) -> None:
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, getattr(UsdPhysics.Tokens, axis))
    # In USD Physics/PhysX, low > high means a locked D6 axis.
    limit_api.CreateLowAttr(1.0)
    limit_api.CreateHighAttr(-1.0)


def create_planar_boom_constraint(
    env: ManagerBasedRLEnv,
    env_ids,
    cfg: BoomConstraintCfg | None = None,
) -> None:
    """Create one world-to-base D6 joint per environment.

    The joint is authored during ``prestartup`` so PhysX parses it when the
    simulation starts. Body0 is omitted, which makes the joint constrain the
    target body relative to the world/static frame.
    """
    del env_ids  # prestartup event; applies to the whole cloned scene.

    if cfg is None:
        cfg = BoomConstraintCfg()

    stage = env.sim.get_initial_stage()
    body_anchor_pos = Gf.Vec3f(*cfg.body_anchor_pos)
    body_anchor_rot = Gf.Quatf(*cfg.body_anchor_rot_wxyz)
    locked_axes = _resolve_locked_axes(cfg)
    created = 0

    for env_id in range(env.num_envs):
        body_path = Sdf.Path(cfg.body_path_template.format(env_id=env_id))
        joint_path = Sdf.Path(cfg.joint_path_template.format(env_id=env_id))
        body_prim = stage.GetPrimAtPath(body_path)

        if not body_prim.IsValid():
            raise RuntimeError(f"Cannot create Forrest boom: body prim does not exist: {body_path}")
        if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Cannot create Forrest boom: target prim is not a rigid body: {body_path}")

        body_tf_w = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        body_tf_w.Orthonormalize()
        world_anchor_pos = body_tf_w.Transform(Gf.Vec3d(*cfg.body_anchor_pos))
        world_anchor_rot = body_tf_w.ExtractRotationQuat()

        joint = UsdPhysics.Joint.Define(stage, joint_path)
        joint.CreateBody1Rel().SetTargets([body_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_anchor_pos))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(world_anchor_rot.real),
                float(world_anchor_rot.imaginary[0]),
                float(world_anchor_rot.imaginary[1]),
                float(world_anchor_rot.imaginary[2]),
            )
        )
        joint.CreateLocalPos1Attr().Set(body_anchor_pos)
        joint.CreateLocalRot1Attr().Set(body_anchor_rot)
        joint.CreateCollisionEnabledAttr(False)

        joint_prim = joint.GetPrim()
        for axis in locked_axes:
            _lock_d6_axis(joint_prim, axis)
        created += 1

    if cfg.debug:
        print(f"[ForrestBoom] Created {created} planar boom D6 joints with locked axes: {locked_axes}.")
