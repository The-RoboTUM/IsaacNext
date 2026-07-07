# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable Forrest tendon simulation helpers.

The standalone tendon runner and PSO runner both need the same robot reset,
controller mapping, and base-constraint logic.  Keeping that logic here avoids
silently diverging conventions between replay and optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.tendons.controllers.base import DOF_ORDER, DOF_SIGN, DOF_TO_ACTUATOR_GROUP, LegControllerBase
from isaaclab.tendons.controllers.cpg import BirdBotCPGLeg, CPGParams, HopfCPGLeg, HopfCPGParams
from isaaclab.tendons.controllers.sinusoidal import SinusoidalLegController, SinusoidalParams


@dataclass(frozen=True)
class ActuatedDofSpec:
    """Resolved mapping from a logical controller DOF to one simulated joint."""

    side: str
    dof: str
    joint_expr: str
    sign: float


def make_actuated_dof_specs(robot_cfg) -> list[ActuatedDofSpec]:
    """Build target joint specs from the configured Forrest actuators."""

    specs: list[ActuatedDofSpec] = []
    for side_prefix, side_name in (("l", "left"), ("r", "right")):
        for dof in DOF_ORDER:
            actuator_group = DOF_TO_ACTUATOR_GROUP[dof]
            actuator_cfg = robot_cfg.actuators[actuator_group]
            matches = [expr for expr in actuator_cfg.joint_names_expr if expr.startswith(side_prefix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {side_name} joint expression for controller DOF {dof!r} in actuator group "
                    f"{actuator_group!r}; got {matches}"
                )
            specs.append(ActuatedDofSpec(side=side_name, dof=dof, joint_expr=matches[0], sign=DOF_SIGN[dof]))
    return specs


def find_actuated_joint_indices(robot, actuated_dof_specs: list[ActuatedDofSpec], *, print_mapping: bool = True):
    """Resolve simulated joint indices for logical controller DOFs."""

    joint_exprs = [spec.joint_expr for spec in actuated_dof_specs]
    joint_indices, found_joint_names = robot.find_joints(joint_exprs, preserve_order=True)
    if len(joint_indices) != len(joint_exprs):
        raise RuntimeError(f"Could not find all actuated joints. Requested: {joint_exprs}; found: {found_joint_names}")

    if print_mapping:
        print("Actuated controller DOFs:")
        for spec, joint_name in zip(actuated_dof_specs, found_joint_names):
            print(f"  {spec.side:>5} {spec.dof:<13} -> {joint_name}")
    return joint_indices


def controller_command_tensor(
    *,
    t: float,
    left_controller: LegControllerBase,
    right_controller: LegControllerBase,
    actuated_dof_specs: list[ActuatedDofSpec],
    initial_joint_positions: torch.Tensor,
    device,
) -> torch.Tensor:
    """Compute one-env controller targets as offsets from the measured initial pose."""

    controllers = {
        "left": left_controller,
        "right": right_controller,
    }

    commands = []
    for spec in actuated_dof_specs:
        q, _qd = controllers[spec.side].joint(spec.dof, t)
        commands.append(spec.sign * q)

    controller_target = torch.tensor([commands], dtype=torch.float32, device=device)
    return initial_joint_positions + controller_target


def make_cpg_legs(params) -> tuple[BirdBotCPGLeg, BirdBotCPGLeg]:
    """Create left/right CPG controllers."""

    common = dict(
        f_hz=params.f_hz,
        D=params.duty_factor,
        A_h_deg=params.hip_amplitude_deg,
        O_h_deg=params.hip_offset_deg,
        A_k_deg=params.knee_amplitude_deg,
        S_f=params.swing_start_offset,
        S_e=params.swing_end_offset,
    )
    left_params = CPGParams(phi0=params.left_phase_offset_rad + params.combined_phase_offset_rad, **common)
    right_params = CPGParams(phi0=params.right_phase_offset_rad + params.combined_phase_offset_rad, **common)
    return (
        BirdBotCPGLeg(left_params, include_knee=params.include_knee),
        BirdBotCPGLeg(right_params, include_knee=params.include_knee),
    )


def make_cpg_oscillator_legs(params) -> tuple[HopfCPGLeg, HopfCPGLeg]:
    """Create left/right phase-locked oscillator CPG controllers."""

    common = dict(
        f_hz=params.f_hz,
        duty_factor=params.duty_factor,
        hip_flexion_amplitude_deg=params.hip_flexion_amplitude_deg,
        hip_flexion_offset_deg=params.hip_flexion_offset_deg,
        hip_flexion_phase_rad=params.hip_flexion_phase_rad,
        knee_flexion_amplitude_deg=params.knee_flexion_amplitude_deg,
        knee_flexion_offset_deg=params.knee_flexion_offset_deg,
        knee_flexion_phase_rad=params.knee_flexion_phase_rad,
        knee_swing_power=params.knee_swing_power,
        hip_roll_amplitude_deg=params.hip_roll_amplitude_deg,
        hip_roll_offset_deg=params.hip_roll_offset_deg,
        hip_roll_phase_rad=params.hip_roll_phase_rad,
        hip_yaw_amplitude_deg=params.hip_yaw_amplitude_deg,
        hip_yaw_offset_deg=params.hip_yaw_offset_deg,
        hip_yaw_phase_rad=params.hip_yaw_phase_rad,
    )
    return (
        HopfCPGLeg(HopfCPGParams(phi0=params.left_phase_rad, **common)),
        HopfCPGLeg(HopfCPGParams(phi0=params.right_phase_rad, **common)),
    )


def make_sinusoidal_legs(params) -> tuple[SinusoidalLegController, SinusoidalLegController]:
    """Create left/right sinusoidal controllers over the same logical DOFs."""

    common = dict(
        f_hz=params.f_hz,
        amplitude_deg=params.amplitude_deg,
        offset_deg=params.offset_deg,
    )
    return (
        SinusoidalLegController(SinusoidalParams(phi0=params.left_phi0_rad, phase_rad=params.left_phase_rad, **common)),
        SinusoidalLegController(
            SinusoidalParams(phi0=params.right_phi0_rad, phase_rad=params.right_phase_rad, **common)
        ),
    )


def make_leg_controllers(run_params) -> tuple[LegControllerBase, LegControllerBase]:
    """Create the configured pair of open-loop leg controllers."""

    if run_params.controller == "cpg":
        return make_cpg_legs(run_params.cpg)
    if run_params.controller == "cpg_oscillator":
        return make_cpg_oscillator_legs(run_params.cpg_oscillator)
    if run_params.controller == "sin":
        return make_sinusoidal_legs(run_params.sinusoidal)
    raise ValueError(f"Unknown controller: {run_params.controller}")


def sinusoidal_command_batch(
    *,
    t: float | torch.Tensor,
    params_by_env: dict[str, torch.Tensor],
    actuated_dof_specs: list[ActuatedDofSpec],
    initial_joint_positions: torch.Tensor,
    controller_zero: torch.Tensor,
) -> torch.Tensor:
    """Vectorized sinusoidal joint target for PSO rollouts.

    ``params_by_env`` stores one value per simulated environment for each
    optimized sinusoidal controller parameter.  Hip roll/yaw stay at their
    configured zero values in the first PSO pass.
    """

    device = initial_joint_positions.device
    dtype = initial_joint_positions.dtype
    f_hz = params_by_env["run.sinusoidal.f_hz"].to(device=device, dtype=dtype)
    t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.expand_as(f_hz)
    omega = 2.0 * torch.pi * f_hz

    commands = []
    for spec in actuated_dof_specs:
        dof = spec.dof
        side = spec.side

        if dof in ("hip_roll", "hip_yaw"):
            q = torch.zeros_like(f_hz)
        else:
            amplitude = torch.deg2rad(
                params_by_env[f"run.sinusoidal.amplitude_deg.{dof}"].to(device=device, dtype=dtype)
            )
            offset = torch.deg2rad(params_by_env[f"run.sinusoidal.offset_deg.{dof}"].to(device=device, dtype=dtype))
            phi0 = params_by_env[f"run.sinusoidal.{side}_phi0_rad"].to(device=device, dtype=dtype)
            phase = params_by_env[f"run.sinusoidal.{side}_phase_rad.{dof}"].to(device=device, dtype=dtype)
            q = amplitude * torch.sin(omega * t_tensor + phi0 + phase) + offset
        commands.append(float(spec.sign) * q)

    controller_target = torch.stack(commands, dim=1)
    return initial_joint_positions + controller_target


def _get_env_parameter(
    params_by_env: dict[str, torch.Tensor],
    name: str,
    default: float,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if name in params_by_env:
        return params_by_env[name].to(device=reference.device, dtype=reference.dtype)
    return torch.full_like(reference, float(default))


def cpg_command_batch(
    *,
    t: float | torch.Tensor,
    params_by_env: dict[str, torch.Tensor],
    actuated_dof_specs: list[ActuatedDofSpec],
    initial_joint_positions: torch.Tensor,
    controller_zero: torch.Tensor,
) -> torch.Tensor:
    """Vectorized basic BirdBot CPG joint target for PSO rollouts."""

    device = initial_joint_positions.device
    dtype = initial_joint_positions.dtype
    num_envs = initial_joint_positions.shape[0]
    reference = torch.ones(num_envs, device=device, dtype=dtype)
    f_hz = _get_env_parameter(params_by_env, "run.cpg.f_hz", 1.0, reference=reference)
    duty_factor = _get_env_parameter(params_by_env, "run.cpg.duty_factor", 0.60, reference=reference).clamp(0.05, 0.95)
    hip_amplitude = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg.hip_amplitude_deg", 20.0, reference=reference)
    )
    hip_offset = torch.deg2rad(_get_env_parameter(params_by_env, "run.cpg.hip_offset_deg", 8.0, reference=reference))
    knee_amplitude = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg.knee_amplitude_deg", 120.0, reference=reference)
    )
    swing_start = _get_env_parameter(params_by_env, "run.cpg.swing_start_offset", 0.02, reference=reference)
    swing_end = _get_env_parameter(params_by_env, "run.cpg.swing_end_offset", 0.05, reference=reference)
    combined_phase = _get_env_parameter(params_by_env, "run.cpg.combined_phase_offset_rad", 0.0, reference=reference)
    left_phase = (
        _get_env_parameter(params_by_env, "run.cpg.left_phase_offset_rad", -torch.pi / 2.0, reference=reference)
        + combined_phase
    )
    right_phase = (
        _get_env_parameter(params_by_env, "run.cpg.right_phase_offset_rad", torch.pi / 2.0, reference=reference)
        + combined_phase
    )

    t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.expand_as(f_hz)
    base_phase = 2.0 * torch.pi * f_hz * t_tensor

    commands = []
    for spec in actuated_dof_specs:
        phase_offset = left_phase if spec.side == "left" else right_phase
        phase = torch.remainder(base_phase + phase_offset, 2.0 * torch.pi)
        stance_phase = phase / (2.0 * duty_factor)
        swing_phase = phase / (2.0 * (1.0 - duty_factor)) + torch.pi * (1.0 - 2.0 * duty_factor) / (1.0 - duty_factor)
        theta = torch.where(phase <= 2.0 * torch.pi * duty_factor, stance_phase, swing_phase)

        if spec.dof == "hip_flexion":
            q = hip_amplitude * torch.sin(theta + torch.pi / 2.0) + hip_offset
        elif spec.dof == "knee_flexion":
            lo = 2.0 * torch.pi * duty_factor + 2.0 * torch.pi * swing_start
            hi = 2.0 * torch.pi - 2.0 * torch.pi * swing_end
            denom = torch.clamp(hi - lo, min=1.0e-6)
            swing_phase = (phase - lo) / denom
            q = torch.where(
                (phase >= lo) & (phase <= hi),
                knee_amplitude * torch.sin(torch.pi * swing_phase),
                torch.zeros_like(phase),
            )
        else:
            q = torch.zeros_like(reference)
        commands.append(float(spec.sign) * q)

    controller_target = torch.stack(commands, dim=1)
    return initial_joint_positions + controller_target


def cpg_oscillator_command_batch(
    *,
    t: float | torch.Tensor,
    params_by_env: dict[str, torch.Tensor],
    actuated_dof_specs: list[ActuatedDofSpec],
    initial_joint_positions: torch.Tensor,
    controller_zero: torch.Tensor,
) -> torch.Tensor:
    """Vectorized phase-locked oscillator CPG target for PSO rollouts."""

    device = initial_joint_positions.device
    dtype = initial_joint_positions.dtype
    num_envs = initial_joint_positions.shape[0]
    reference = torch.ones(num_envs, device=device, dtype=dtype)
    f_hz = _get_env_parameter(params_by_env, "run.cpg_oscillator.f_hz", 0.8, reference=reference)
    duty_factor = _get_env_parameter(params_by_env, "run.cpg_oscillator.duty_factor", 0.60, reference=reference).clamp(
        0.05, 0.95
    )
    t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.expand_as(f_hz)
    omega = 2.0 * torch.pi * f_hz

    base_phase = omega * t_tensor

    hip_flexion_amplitude = torch.deg2rad(
        _get_env_parameter(
            params_by_env,
            "run.cpg_oscillator.hip_flexion_amplitude_deg",
            24.0,
            reference=reference,
        )
    )
    hip_flexion_offset = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg_oscillator.hip_flexion_offset_deg", 8.0, reference=reference)
    )
    hip_flexion_phase = _get_env_parameter(
        params_by_env,
        "run.cpg_oscillator.hip_flexion_phase_rad",
        0.0,
        reference=reference,
    )
    knee_flexion_amplitude = torch.deg2rad(
        _get_env_parameter(
            params_by_env,
            "run.cpg_oscillator.knee_flexion_amplitude_deg",
            34.0,
            reference=reference,
        )
    )
    knee_flexion_offset = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg_oscillator.knee_flexion_offset_deg", 0.0, reference=reference)
    )
    knee_flexion_phase = _get_env_parameter(
        params_by_env,
        "run.cpg_oscillator.knee_flexion_phase_rad",
        torch.pi / 2.0,
        reference=reference,
    )
    knee_swing_power = _get_env_parameter(
        params_by_env,
        "run.cpg_oscillator.knee_swing_power",
        1.5,
        reference=reference,
    ).clamp(0.1, 6.0)

    hip_roll_amplitude = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg_oscillator.hip_roll_amplitude_deg", 0.0, reference=reference)
    )
    hip_roll_offset = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg_oscillator.hip_roll_offset_deg", 0.0, reference=reference)
    )
    hip_roll_phase = _get_env_parameter(
        params_by_env, "run.cpg_oscillator.hip_roll_phase_rad", 0.0, reference=reference
    )
    hip_yaw_amplitude = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg_oscillator.hip_yaw_amplitude_deg", 0.0, reference=reference)
    )
    hip_yaw_offset = torch.deg2rad(
        _get_env_parameter(params_by_env, "run.cpg_oscillator.hip_yaw_offset_deg", 0.0, reference=reference)
    )
    hip_yaw_phase = _get_env_parameter(params_by_env, "run.cpg_oscillator.hip_yaw_phase_rad", 0.0, reference=reference)

    commands = []
    for spec in actuated_dof_specs:
        phase_offset = _get_env_parameter(
            params_by_env,
            f"run.cpg_oscillator.{spec.side}_phase_rad",
            0.0 if spec.side == "left" else torch.pi,
            reference=reference,
        )
        phase = torch.remainder(base_phase + phase_offset, 2.0 * torch.pi)
        stance_phase = phase / (2.0 * duty_factor)
        swing_phase = phase / (2.0 * (1.0 - duty_factor)) + torch.pi * (1.0 - 2.0 * duty_factor) / (1.0 - duty_factor)
        theta = torch.where(phase <= 2.0 * torch.pi * duty_factor, stance_phase, swing_phase)

        if spec.dof == "hip_roll":
            q = hip_roll_amplitude * torch.sin(theta + hip_roll_phase) + hip_roll_offset
        elif spec.dof == "hip_yaw":
            q = hip_yaw_amplitude * torch.sin(theta + hip_yaw_phase) + hip_yaw_offset
        elif spec.dof == "hip_flexion":
            q = hip_flexion_amplitude * torch.sin(theta + hip_flexion_phase) + hip_flexion_offset
        elif spec.dof == "knee_flexion":
            swing = torch.clamp(torch.sin(theta + knee_flexion_phase), min=0.0)
            q = knee_flexion_offset + knee_flexion_amplitude * torch.pow(swing, knee_swing_power)
        else:
            q = torch.zeros_like(reference)
        commands.append(float(spec.sign) * q)

    controller_target = torch.stack(commands, dim=1)
    return initial_joint_positions + controller_target


def open_loop_command_batch(
    *,
    t: float | torch.Tensor,
    params_by_env: dict[str, torch.Tensor],
    actuated_dof_specs: list[ActuatedDofSpec],
    initial_joint_positions: torch.Tensor,
    controller_zero: torch.Tensor,
) -> torch.Tensor:
    """Dispatch vectorized open-loop controller targets based on optimized parameter names."""

    if any(name.startswith("run.cpg_oscillator.") for name in params_by_env):
        return cpg_oscillator_command_batch(
            t=t,
            params_by_env=params_by_env,
            actuated_dof_specs=actuated_dof_specs,
            initial_joint_positions=initial_joint_positions,
            controller_zero=controller_zero,
        )
    if any(name.startswith("run.cpg.") for name in params_by_env):
        return cpg_command_batch(
            t=t,
            params_by_env=params_by_env,
            actuated_dof_specs=actuated_dof_specs,
            initial_joint_positions=initial_joint_positions,
            controller_zero=controller_zero,
        )
    return sinusoidal_command_batch(
        t=t,
        params_by_env=params_by_env,
        actuated_dof_specs=actuated_dof_specs,
        initial_joint_positions=initial_joint_positions,
        controller_zero=controller_zero,
    )


def reset_robot_to_default(
    robot,
    *,
    env_origins: torch.Tensor | None = None,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Push configured default state into PhysX and reset internal articulation buffers."""

    if env_ids is None:
        env_indices = slice(None)
        selected_origins = env_origins
    else:
        env_indices = env_ids
        selected_origins = None if env_origins is None else env_origins[env_ids]

    root_state = robot.data.default_root_state[env_indices].clone()
    if env_origins is not None:
        root_state[:, :3] += selected_origins
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        position=robot.data.default_joint_pos[env_indices],
        velocity=robot.data.default_joint_vel[env_indices],
        env_ids=env_ids,
    )
    robot.set_joint_position_target(robot.data.default_joint_pos[env_indices], env_ids=env_ids)
    robot.write_data_to_sim()
    robot.reset(env_ids)


def resolve_boom_locked_axes(params) -> tuple[str, ...]:
    """Return the configured boom D6 axes, including the optional sagittal angle lock."""

    locked_axes = tuple(params.boom.locked_axes)
    pitch_axis = "rotY" if "transY" in locked_axes else "rotX"
    if params.boom.lock_x_angle and pitch_axis not in locked_axes:
        return (*locked_axes, pitch_axis)
    return locked_axes


def lock_d6_axis(joint_prim, axis: str) -> None:
    """Author a locked USD Physics D6 limit axis."""

    from pxr import UsdPhysics

    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, getattr(UsdPhysics.Tokens, axis))
    # In USD Physics/PhysX, low > high means a locked D6 axis.
    limit_api.CreateLowAttr(1.0)
    limit_api.CreateHighAttr(-1.0)


def add_fixed_world_joint(sim, params, *, body_path: str | None = None, joint_path: str | None = None):
    """Lock a Forrest base body to the world with a fixed USD joint."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = sim.stage
    body_path = body_path or params.robot.fixed_world_body_path
    joint_path = joint_path or params.robot.fixed_world_joint_path
    body_sdf_path = Sdf.Path(body_path)
    joint_sdf_path = Sdf.Path(joint_path)
    body_prim = stage.GetPrimAtPath(body_sdf_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"Cannot create fixed world joint: body prim does not exist: {body_sdf_path}")

    body_tf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_tf.Orthonormalize()
    body_pos_w = body_tf.ExtractTranslation()
    body_rot_w = body_tf.ExtractRotationQuat()

    if params.robot.fixed_world_joint_local_pos0 is None:
        local_pos0 = Gf.Vec3f(float(body_pos_w[0]), float(body_pos_w[1]), float(body_pos_w[2]))
    else:
        local_pos0 = Gf.Vec3f(*params.robot.fixed_world_joint_local_pos0)

    if params.robot.fixed_world_joint_local_rot0_wxyz is None:
        local_rot0 = Gf.Quatf(
            float(body_rot_w.real),
            float(body_rot_w.imaginary[0]),
            float(body_rot_w.imaginary[1]),
            float(body_rot_w.imaginary[2]),
        )
    else:
        local_rot0 = Gf.Quatf(*params.robot.fixed_world_joint_local_rot0_wxyz)

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_sdf_path)
    fixed_joint.CreateBody1Rel().SetTargets([body_sdf_path])
    fixed_joint.CreateLocalPos0Attr(local_pos0)
    fixed_joint.CreateLocalRot0Attr(local_rot0)
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateCollisionEnabledAttr(False)


def add_planar_boom_joint(sim, params, *, body_path: str | None = None, joint_path: str | None = None):
    """Constrain a Forrest base body to the configured sagittal plane with a D6 joint."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = sim.stage
    body_sdf_path = Sdf.Path(body_path or params.robot.fixed_world_body_path)
    joint_sdf_path = Sdf.Path(joint_path or params.robot.fixed_world_joint_path + "_planar_boom")
    body_prim = stage.GetPrimAtPath(body_sdf_path)

    if not body_prim.IsValid():
        raise RuntimeError(f"Cannot create Forrest boom: body prim does not exist: {body_sdf_path}")
    if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Cannot create Forrest boom: target prim is not a rigid body: {body_sdf_path}")

    body_anchor_pos = Gf.Vec3f(*params.boom.body_anchor_pos)
    body_anchor_rot = Gf.Quatf(*params.boom.body_anchor_rot_wxyz)
    body_tf_w = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_tf_w.Orthonormalize()
    world_anchor_pos = body_tf_w.Transform(Gf.Vec3d(*params.boom.body_anchor_pos))
    world_anchor_rot = body_tf_w.ExtractRotationQuat()

    joint = UsdPhysics.Joint.Define(stage, joint_sdf_path)
    joint.CreateBody1Rel().SetTargets([body_sdf_path])
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

    locked_axes = resolve_boom_locked_axes(params)
    for axis in locked_axes:
        lock_d6_axis(joint.GetPrim(), axis)

    if params.boom.debug:
        print(f"[ForrestBoom] Created planar boom D6 joint at {joint_sdf_path} with locked axes: {locked_axes}.")


def configure_base_constraint(sim, params, constraint_mode: str) -> None:
    """Author the selected standalone base constraint before PhysX startup."""

    if constraint_mode == "freefall":
        return
    if constraint_mode == "boom":
        add_planar_boom_joint(sim, params)
        return
    if constraint_mode == "static":
        add_fixed_world_joint(sim, params)
        return
    if constraint_mode == "static_boom":
        # This intentionally authors both constraints for diagnostics. The fixed joint fully locks the base,
        # so the planar boom does not add motion freedom unless the fixed joint is later relaxed.
        add_fixed_world_joint(sim, params)
        add_planar_boom_joint(sim, params)
        return
    raise ValueError(f"Unknown constraint_mode: {constraint_mode!r}")


def configure_scene_base_constraints(sim, params, constraint_mode: str, num_envs: int) -> None:
    """Author per-env base constraints for cloned Forrest environments."""

    if constraint_mode == "freefall":
        return
    for env_id in range(num_envs):
        body_path = params.boom.body_path_template.format(env_id=env_id)
        if constraint_mode == "boom":
            joint_path = params.boom.joint_path_template.format(env_id=env_id)
            add_planar_boom_joint(sim, params, body_path=body_path, joint_path=joint_path)
        elif constraint_mode == "static":
            joint_path = f"{body_path}_fixed_joint"
            add_fixed_world_joint(sim, params, body_path=body_path, joint_path=joint_path)
        elif constraint_mode == "static_boom":
            fixed_joint_path = f"{body_path}_fixed_joint"
            boom_joint_path = params.boom.joint_path_template.format(env_id=env_id)
            add_fixed_world_joint(sim, params, body_path=body_path, joint_path=fixed_joint_path)
            add_planar_boom_joint(sim, params, body_path=body_path, joint_path=boom_joint_path)
        else:
            raise ValueError(f"Unknown constraint_mode: {constraint_mode!r}")


def set_tendon_lengths_by_env(
    tendon_data: Any,
    params_by_env: dict[str, torch.Tensor],
    *,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Overwrite optimized tendon length/stiffness tensors in a batched ``TendonData`` object.

    ``TendonData`` stores all left-leg rows first and all right-leg rows second.
    A value tensor shaped ``(num_envs,)`` therefore maps to ``cat([values, values])``.
    """

    tendon_attrs = {
        "tendons.baseline.lengths.gst_spring_rest": "gst_spring_rest_length",
        "tendons.baseline.lengths.upper_gst": "upper_gst_length",
        "tendons.baseline.lengths.lower_gst": "lower_gst_length",
        "tendons.baseline.lengths.dft": "dft_length",
        "tendons.baseline.lengths.edt1": "edt1_length",
        "tendons.baseline.lengths.edt2": "edt2_length",
        "tendons.baseline.lengths.kft": "kft_length",
        "tendons.baseline.stiffness.gst": "gst_stiffness",
        "tendons.baseline.stiffness.dft": "dft_stiffness",
        "tendons.baseline.stiffness.edt1": "edt1_stiffness",
        "tendons.baseline.stiffness.edt2": "edt2_stiffness",
        "tendons.baseline.stiffness.kft": "kft_stiffness",
    }
    for param_name, attr_name in tendon_attrs.items():
        if param_name not in params_by_env:
            continue
        current = getattr(tendon_data, attr_name)
        values = params_by_env[param_name].to(device=current.device, dtype=current.dtype)
        num_envs = current.shape[0] // 2
        if env_ids is None:
            current[:num_envs] = values
            current[num_envs:] = values
        else:
            current[env_ids] = values
            current[env_ids + num_envs] = values
