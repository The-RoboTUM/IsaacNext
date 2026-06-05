# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

# Experimental
from isaaclab.tendons.models.analytic.constants import actuated_joint_names
from isaaclab.tendons.parameter_loader import load_forrest_parameter_config
from isaaclab.tendons.plugin.action_term_cfg import TendonActionTermHybridCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

# -----------------------------------------------------------------------------
# Centralized Forrest parameters
# -----------------------------------------------------------------------------
FORREST_PARAMS = load_forrest_parameter_config()
FORREST_TENDON_RANDOMIZATION = FORREST_PARAMS.to_tendon_randomization_ranges()
REWARD_WEIGHTS = FORREST_PARAMS.training.rewards.weights
TRAINING_PARAMS = FORREST_PARAMS.training
CONTACT_PARAMS = TRAINING_PARAMS.contacts
REWARD_PARAMS = TRAINING_PARAMS.rewards

FEET_CFG = SceneEntityCfg(
    "robot",
    body_names=CONTACT_PARAMS.foot_body_names,
)

FOOT_CONNECTOR_CFG = SceneEntityCfg(
    "robot",
    body_names=CONTACT_PARAMS.foot_connector_body_names,
)

# Foot order in FEET_CFG. Keep these as parameters so swapping bodies later is easy.
#   s45_digit_assy_1 = right foot
#   s45_digit_assy_2 = left foot
RIGHT_FOOT_INDEX = CONTACT_PARAMS.right_foot_index
LEFT_FOOT_INDEX = CONTACT_PARAMS.left_foot_index

# Foot reward frame convention.
# Current model:
#   forward = -Y  -> (0, -1, 0)
#   lateral = +X  -> (1,  0, 0)
# After fixing the USD so robot +X is front, change to:
#   FEET_FORWARD_DIR_B = (1.0, 0.0, 0.0)
#   FEET_LATERAL_DIR_B = (0.0, 1.0, 0.0)
FEET_FORWARD_DIR_B = CONTACT_PARAMS.forward_dir_b
FEET_LATERAL_DIR_B = CONTACT_PARAMS.lateral_dir_b

# Kept for compatibility with older code/comments.
FORWARD_AXIS = 1
FORWARD_SIGN = -1.0
LATERAL_AXIS = 0
LEFT_SIGN = 1.0


def _body_name_regex(body_names: tuple[str, ...] | list[str]) -> str:
    return "(" + "|".join(body_names) + ")"


def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab quaternions from (w, x, y, z) to rotation matrices.

    Kept with the original function name so existing code can call it safely.
    Isaac Lab stores body/root quaternions in wxyz order.
    """
    return quat_to_rot_matrix_wxyz(q)


def quat_to_rot_matrix_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab quaternions from (w, x, y, z) to rotation matrices.

    Supports shape:
        [N, 4]
        [N, B, 4]
    Returns:
        [N, 3, 3] or [N, B, 3, 3]
    """
    w, x, y, z = q.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    )

    return rot.reshape(*q.shape[:-1], 3, 3)


def _unit_vec(
    values: tuple[float, float, float],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a finite unit vector on the requested device."""
    vec = torch.tensor(values, device=device, dtype=dtype)
    norm = torch.norm(vec).clamp(min=1e-8)
    return torch.nan_to_num(vec / norm, nan=0.0, posinf=0.0, neginf=0.0)


def _project_along(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project [..., 3] vectors onto a unit direction."""
    return torch.sum(values * direction, dim=-1)


def _safe_nonnegative_reward(value: torch.Tensor, max_value: float | None = None) -> torch.Tensor:
    """Make custom penalty terms finite so they cannot poison PPO."""
    if max_value is None:
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(value, min=0.0)

    value = torch.nan_to_num(value, nan=0.0, posinf=max_value, neginf=0.0)
    return torch.clamp(value, min=0.0, max=max_value)


def _debug_index(env: ManagerBasedRLEnv, debug_env_id: int) -> int:
    """Clamp debug env index so prints cannot crash small play runs."""
    return int(max(0, min(debug_env_id, env.num_envs - 1)))


def reset_root_state_uniform_all_envs_on_startup(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Startup-safe wrapper around Isaac Lab's root reset event.

    Startup events receive ``env_ids=None`` from the event manager, while
    ``reset_root_state_uniform`` expects a tensor. For startup, apply the reset
    to every environment.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    mdp.reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)


def feet_crossing_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = FEET_CFG,
    min_lateral_separation: float = 0.03,
    lateral_dir_b: tuple[float, float, float] = FEET_LATERAL_DIR_B,
    expected_foot0_lateral_order: float = -1.0,
    side_margin: float = 0.02,
    max_crossing_error: float = 0.25,
    debug: bool = False,
    debug_every: int = 100,
    debug_env_id: int = 0,
) -> torch.Tensor:
    """Penalize feet crossing using explicit foot-0/foot-1 lateral ordering.

    This avoids the previous left/right-index ambiguity.

    ``expected_foot0_lateral_order`` means:
        -1.0: foot 0 should be on the negative side of foot 1 along ``lateral_dir_b``.
              With current settings this means foot 0 should have smaller +X than foot 1.
        +1.0: foot 0 should be on the positive side of foot 1 along ``lateral_dir_b``.

    For your debug logs, crossed poses had foot_0_lateral > foot_1_lateral, so the
    correct setting is the default: ``expected_foot0_lateral_order = -1.0``.
    """
    pos_b, _ = get_feet_pose_base(env, asset_cfg)

    lateral_dir = _unit_vec(lateral_dir_b, env.device)
    lateral_coord = _project_along(pos_b, lateral_dir)  # [num_envs, 2]

    # Make this a scalar tensor so all math stays on device.
    order_sign = torch.tensor(
        1.0 if expected_foot0_lateral_order >= 0.0 else -1.0,
        device=env.device,
        dtype=torch.float32,
    )

    # Positive signed_separation means the feet are in the desired order.
    # For order_sign = -1: desired is foot0 < foot1.
    # For order_sign = +1: desired is foot0 > foot1.
    delta_01 = lateral_coord[:, 0] - lateral_coord[:, 1]
    signed_separation = order_sign * delta_01

    order_error = torch.clamp(
        min_lateral_separation - signed_separation,
        min=0.0,
        max=max_crossing_error,
    )

    # Extra centerline term: foot 0 and foot 1 should stay on their expected sides
    # of the body, not only maintain relative ordering. This makes crossing much
    # harder to game.
    desired_side_signs = torch.stack((order_sign, -order_sign))  # [2]
    side_score = lateral_coord * desired_side_signs[None, :]
    side_error = torch.clamp(
        side_margin - side_score,
        min=0.0,
        max=max_crossing_error,
    )

    penalty = order_error.square() + 0.5 * side_error.square().mean(dim=1)
    penalty = _safe_nonnegative_reward(penalty, max_crossing_error**2 * 1.5)

    if debug and hasattr(env, "common_step_counter") and debug_every > 0:
        if env.common_step_counter % debug_every == 0:
            i = _debug_index(env, debug_env_id)
            print("\n[feet_crossing_penalty]")
            print(f"  step: {env.common_step_counter}")
            print(f"  lateral_dir_b: {lateral_dir_b}")
            print(f"  expected_foot0_lateral_order: {expected_foot0_lateral_order}")
            print(f"  foot_0_pos_b: {pos_b[i, 0].detach().cpu().numpy()}")
            print(f"  foot_1_pos_b: {pos_b[i, 1].detach().cpu().numpy()}")
            print(f"  foot_0_lateral: {lateral_coord[i, 0].item():.4f}")
            print(f"  foot_1_lateral: {lateral_coord[i, 1].item():.4f}")
            print(f"  delta_01: {delta_01[i].item():.4f}")
            print(f"  signed_separation_good_if_positive: {signed_separation[i].item():.4f}")
            print(f"  order_error: {order_error[i].item():.4f}")
            print(f"  side_score: {side_score[i].detach().cpu().numpy()}")
            print(f"  side_error: {side_error[i].detach().cpu().numpy()}")
            print(f"  penalty_unweighted: {penalty[i].item():.6f}")

    return penalty


def feet_parallel_contact_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = FEET_CFG,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=FEET_CFG.body_names),
    contact_threshold: float = 1.0,
    sole_normal_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ground_normal_w: tuple[float, float, float] = (0.0, 0.0, 1.0),
    debug: bool = False,
    debug_every: int = 100,
    debug_env_id: int = 0,
) -> torch.Tensor:
    """Penalize foot-ground contact when the foot sole is tilted.

    The only frame-specific parameter is ``sole_normal_axis``: the local foot
    axis normal to the sole. The sign does not matter because the dot product is
    absolute-valued.
    """
    robot = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]

    foot_ids = asset_cfg.body_ids
    foot_quat_w = robot.data.body_quat_w[:, foot_ids, :]
    foot_quat_w = foot_quat_w / torch.norm(foot_quat_w, dim=-1, keepdim=True).clamp(min=1e-8)

    foot_rot_w = quat_to_rot_matrix_wxyz(foot_quat_w)

    local_axis = _unit_vec(sole_normal_axis, env.device)
    sole_normal_w = torch.einsum("nbij,j->nbi", foot_rot_w, local_axis)
    sole_normal_w = torch.nan_to_num(sole_normal_w, nan=0.0, posinf=0.0, neginf=0.0)

    ground_normal = _unit_vec(ground_normal_w, env.device)

    cos_angle = torch.sum(sole_normal_w * ground_normal, dim=-1).abs()
    cos_angle = torch.nan_to_num(cos_angle, nan=0.0, posinf=1.0, neginf=0.0)
    cos_angle = torch.clamp(cos_angle, 0.0, 1.0)

    angle_error = torch.acos(cos_angle)  # [num_envs, 2], in [0, pi/2] because of abs()

    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact_norm = torch.norm(contact_force, dim=-1)
    contact_norm = torch.nan_to_num(contact_norm, nan=0.0, posinf=0.0, neginf=0.0)
    in_contact = contact_norm > contact_threshold

    penalty_per_foot = angle_error.square() * in_contact.float()
    num_contacts = in_contact.float().sum(dim=1).clamp(min=1.0)
    penalty = penalty_per_foot.sum(dim=1) / num_contacts
    penalty = _safe_nonnegative_reward(penalty, (math.pi / 2.0) ** 2)

    if debug and hasattr(env, "common_step_counter") and debug_every > 0:
        if env.common_step_counter % debug_every == 0:
            i = _debug_index(env, debug_env_id)
            print("\n[feet_parallel_contact_penalty]")
            print(f"  step: {env.common_step_counter}")
            print(f"  sole_normal_axis: {sole_normal_axis}")
            print(f"  ground_normal_w: {ground_normal_w}")
            print(f"  contact_norm: {contact_norm[i].detach().cpu().numpy()}")
            print(f"  in_contact: {in_contact[i].detach().cpu().numpy()}")
            print(f"  sole_normal_w foot 0: {sole_normal_w[i, 0].detach().cpu().numpy()}")
            print(f"  sole_normal_w foot 1: {sole_normal_w[i, 1].detach().cpu().numpy()}")
            print(f"  angle_error_deg: {torch.rad2deg(angle_error[i]).detach().cpu().numpy()}")
            print(f"  penalty_unweighted: {penalty[i].item():.6f}")

    return penalty


def get_feet_pose_base(env, feet_cfg: SceneEntityCfg = FEET_CFG):
    robot = env.scene[feet_cfg.name]
    ids = feet_cfg.body_ids

    pos_w = robot.data.body_pos_w[:, ids, :]
    quat_w = robot.data.body_quat_w[:, ids, :]

    base_pos = robot.data.root_pos_w[:, None, :]
    base_quat = robot.data.root_quat_w
    base_quat = base_quat / torch.norm(base_quat, dim=-1, keepdim=True).clamp(min=1e-8)

    rel = pos_w - base_pos

    rot = quat_to_rot_matrix(base_quat)
    rel_b = torch.einsum("nij,nkj->nki", rot.transpose(1, 2), rel)
    rel_b = torch.nan_to_num(rel_b, nan=0.0, posinf=0.0, neginf=0.0)

    return rel_b, quat_w


def feet_symmetry_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = FEET_CFG,
    alpha: float = 0.01,  # kept for backward compatibility; unused by this direct penalty
    forward_dir_b: tuple[float, float, float] = FEET_FORWARD_DIR_B,
    max_forward_separation: float = 0.25,
    max_forward_error: float = 0.75,
    debug: bool = False,
    debug_every: int = 100,
    debug_env_id: int = 0,
) -> torch.Tensor:
    """Penalize excessive fore/aft foot split.

    The old EMA "which foot is ahead" term stayed near zero for short rollouts
    and did not push against the observed static one-leg-ahead posture. This
    direct term is still order-agnostic, but immediately penalizes excessive
    separation along the robot forward axis.
    """
    pos_b, _ = get_feet_pose_base(env, asset_cfg)

    forward_dir = _unit_vec(forward_dir_b, env.device)
    forward_coord = _project_along(pos_b, forward_dir)  # [num_envs, 2]

    forward_separation = torch.abs(forward_coord[:, 0] - forward_coord[:, 1])
    forward_error = torch.clamp(
        forward_separation - max_forward_separation,
        min=0.0,
        max=max_forward_error,
    )
    penalty = _safe_nonnegative_reward(forward_error.square(), max_forward_error**2)

    if debug and hasattr(env, "common_step_counter") and debug_every > 0:
        if env.common_step_counter % debug_every == 0:
            i = _debug_index(env, debug_env_id)
            print("\n[feet_symmetry_penalty]")
            print(f"  step: {env.common_step_counter}")
            print(f"  forward_dir_b: {forward_dir_b}")
            print(f"  foot_0_pos_b: {pos_b[i, 0].detach().cpu().numpy()}")
            print(f"  foot_1_pos_b: {pos_b[i, 1].detach().cpu().numpy()}")
            print(f"  foot_0_forward: {forward_coord[i, 0].item():.4f}")
            print(f"  foot_1_forward: {forward_coord[i, 1].item():.4f}")
            print(f"  forward_separation: {forward_separation[i].item():.4f}")
            print(f"  max_forward_separation: {max_forward_separation:.4f}")
            print(f"  forward_error: {forward_error[i].item():.4f}")
            print(f"  penalty_unweighted: {penalty[i].item():.6f}")

    return penalty


def terminate_if_base_too_low(env, minimum_height: float = 0.8):
    # Torch tensor: (num_envs, num_bodies, 3)
    body_pos = env.scene["robot"].data.body_pos_w

    # z-coordinate of base body (index 0 or use name lookup)
    base_z = body_pos[:, 0, 2]  # shape (num_envs,)

    # return a torch.BoolTensor mask
    return base_z < minimum_height


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float = 1.4,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for keeping its base/root height near target_height.

    Returns a reward in [0, 1]:
      - 1.0 when base height is exactly target_height
      - smaller when the base is too low or too high
    """
    robot = env.scene[asset_cfg.name]

    # Root/base z position in world frame.
    # Shape: [num_envs]
    base_z = robot.data.root_pos_w[:, 2]

    height_error = base_z - target_height

    return torch.exp(-height_error.square() / std**2)


@configclass
class ForrestRewards(RewardsCfg):
    # Reward terms for the MDP.

    feet_crossing = RewTerm(
        func=feet_crossing_penalty,
        weight=REWARD_WEIGHTS["feet_crossing"],
        params={
            "asset_cfg": FEET_CFG,
            "min_lateral_separation": REWARD_PARAMS.feet_crossing["min_lateral_separation"],
            "lateral_dir_b": FEET_LATERAL_DIR_B,
            "expected_foot0_lateral_order": REWARD_PARAMS.feet_crossing["expected_foot0_lateral_order"],
            "side_margin": REWARD_PARAMS.feet_crossing["side_margin"],
            "max_crossing_error": REWARD_PARAMS.feet_crossing["max_crossing_error"],
            "debug": REWARD_PARAMS.feet_crossing["debug"],
            "debug_every": REWARD_PARAMS.feet_crossing["debug_every"],
            "debug_env_id": REWARD_PARAMS.feet_crossing["debug_env_id"],
        },
    )

    feet_parallel_contact = RewTerm(
        func=feet_parallel_contact_penalty,
        weight=REWARD_WEIGHTS["feet_parallel_contact"],
        params={
            "asset_cfg": FEET_CFG,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_CFG.body_names),
            "contact_threshold": REWARD_PARAMS.feet_parallel_contact["contact_threshold"],
            "sole_normal_axis": REWARD_PARAMS.feet_parallel_contact["sole_normal_axis"],
            "ground_normal_w": REWARD_PARAMS.feet_parallel_contact["ground_normal_w"],
            "debug": REWARD_PARAMS.feet_parallel_contact["debug"],
            "debug_every": REWARD_PARAMS.feet_parallel_contact["debug_every"],
            "debug_env_id": REWARD_PARAMS.feet_parallel_contact["debug_env_id"],
        },
    )

    track_base_height_exp = RewTerm(
        func=track_base_height_exp,
        weight=REWARD_WEIGHTS["track_base_height_exp"],
        params={
            "target_height": REWARD_PARAMS.track_base_height["target_height"],
            "std": REWARD_PARAMS.track_base_height["std"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    alive = RewTerm(func=mdp.is_alive, weight=REWARD_WEIGHTS["alive"])

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=REWARD_WEIGHTS["termination_penalty"],
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=REWARD_WEIGHTS["track_lin_vel_xy_exp"],
        params={"command_name": "base_velocity", "std": REWARD_PARAMS.track_velocity["lin_vel_xy_std"]},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=REWARD_WEIGHTS["track_ang_vel_z_exp"],
        params={"command_name": "base_velocity", "std": REWARD_PARAMS.track_velocity["ang_vel_z_std"]},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=REWARD_WEIGHTS["feet_air_time"],
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_CFG.body_names),
            "threshold": REWARD_PARAMS.feet_air_time_threshold,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=REWARD_WEIGHTS["feet_slide"],
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_CFG.body_names),
            "asset_cfg": FEET_CFG,
        },
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=REWARD_WEIGHTS["joint_deviation_l1"],
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l1_acetabulofemoral_lateral",
                    "r1_acetabulofemoral_lateral",
                ],
            )
        },
    )

    hip_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=REWARD_WEIGHTS["hip_deviation_l1"],
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l0_acetabulofemoral_roll",
                    "r0_acetabulofemoral_roll",
                ],
            )
        },
    )

    gait_symmetry = RewTerm(
        func=feet_symmetry_penalty,
        weight=REWARD_WEIGHTS["gait_symmetry"],
        params={
            "asset_cfg": FEET_CFG,
            "forward_dir_b": FEET_FORWARD_DIR_B,
            "max_forward_separation": REWARD_PARAMS.gait_symmetry["max_forward_separation"],
            "max_forward_error": REWARD_PARAMS.gait_symmetry["max_forward_error"],
            "debug": REWARD_PARAMS.gait_symmetry["debug"],
            "debug_every": REWARD_PARAMS.gait_symmetry["debug_every"],
            "debug_env_id": REWARD_PARAMS.gait_symmetry["debug_env_id"],
        },
    )

    foot_connector_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=REWARD_WEIGHTS["foot_connector_contact"],
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_CONNECTOR_CFG.body_names),
            "threshold": REWARD_PARAMS.undesired_contact_threshold,
        },
    )


@configclass
class ForrestActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=actuated_joint_names,
        # scale={
        #     ".*roll": math.radians(15),
        #     ".*lateral": math.radians(15),
        #     ".*flexion": math.radians(20),
        #     ".*flexor": math.radians(75),
        # },
        scale={
            joint_expr: math.radians(scale_deg) for joint_expr, scale_deg in TRAINING_PARAMS.actions.scale_deg.items()
        },
        use_default_offset=TRAINING_PARAMS.actions.use_default_offset,
    )

    tendon = TendonActionTermHybridCfg(
        asset_name="robot",
        randomization_ranges=FORREST_TENDON_RANDOMIZATION,
        parameters_file=None,
    )
