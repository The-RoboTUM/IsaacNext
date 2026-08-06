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
#   forward = +X -> (1, 0, 0)
#   lateral = +Y -> (0, 1, 0)
FEET_FORWARD_DIR_B = CONTACT_PARAMS.forward_dir_b
FEET_LATERAL_DIR_B = CONTACT_PARAMS.lateral_dir_b

# Kept for compatibility with older code/comments.
FORWARD_AXIS = 0
FORWARD_SIGN = 1.0
LATERAL_AXIS = 1
LEFT_SIGN = 1.0


def _body_name_regex(body_names: tuple[str, ...] | list[str]) -> str:
    return "(" + "|".join(body_names) + ")"


def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab quaternions from (w, x, y, z) to rotation matrices.

    Kept with the original function name so existing code can call it safely.
    Isaac Lab stores body/root quaternions in wxyz order.
    """
    return quat_to_rot_matrix_wxyz(q)


@torch.jit.script
def quat_to_rot_matrix_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab quaternions from (w, x, y, z) to rotation matrices.

    TorchScript-friendly for shape [N, 4] or [N, B, 4].
    """
    w, x, y, z = q.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    )

    if q.dim() == 2:
        return rot.reshape(q.size(0), 3, 3)
    return rot.reshape(q.size(0), q.size(1), 3, 3)


def _unit_vec(
    values: tuple[float, float, float],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a finite unit vector on the requested device."""
    vec = torch.tensor(values, device=device, dtype=dtype)
    norm = torch.norm(vec).clamp(min=1e-8)
    return torch.nan_to_num(vec / norm, nan=0.0, posinf=0.0, neginf=0.0)


@torch.jit.script
def _normalize_vec3_tensor(vec: torch.Tensor) -> torch.Tensor:
    """Return a finite unit vector. Input must already be on the target device/dtype."""
    norm = torch.norm(vec).clamp(min=1e-8)
    return torch.nan_to_num(vec / norm, nan=0.0, posinf=0.0, neginf=0.0)


@torch.jit.script
def _project_along(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project [..., 3] vectors onto a unit direction."""
    return torch.sum(values * direction, dim=-1)


@torch.jit.script
def _safe_nonnegative_reward_unbounded(value: torch.Tensor) -> torch.Tensor:
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(value, min=0.0)


@torch.jit.script
def _safe_nonnegative_reward_bounded(value: torch.Tensor, max_value: float) -> torch.Tensor:
    value = torch.nan_to_num(value, nan=0.0, posinf=max_value, neginf=0.0)
    return torch.clamp(value, min=0.0, max=max_value)


def _safe_nonnegative_reward(value: torch.Tensor, max_value: float | None = None) -> torch.Tensor:
    """Make custom penalty terms finite so they cannot poison PPO."""
    if max_value is None:
        return _safe_nonnegative_reward_unbounded(value)
    return _safe_nonnegative_reward_bounded(value, max_value)


def finite_observation(data: torch.Tensor) -> torch.Tensor:
    """Replace non-finite policy observations before they enter the rollout buffer."""
    return torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


@torch.jit.script
def _feet_crossing_penalty_core(
    pos_b: torch.Tensor,
    lateral_dir: torch.Tensor,
    min_lateral_separation: float,
    expected_foot0_lateral_order: float,
    side_margin: float,
    max_crossing_error: float,
) -> torch.Tensor:
    lateral_dir = _normalize_vec3_tensor(lateral_dir)
    lateral_coord = _project_along(pos_b, lateral_dir)

    order_sign = -1.0
    if expected_foot0_lateral_order >= 0.0:
        order_sign = 1.0

    delta_01 = lateral_coord[:, 0] - lateral_coord[:, 1]
    signed_separation = order_sign * delta_01

    order_error = torch.clamp(
        min_lateral_separation - signed_separation,
        min=0.0,
        max=max_crossing_error,
    )

    side_score_0 = lateral_coord[:, 0] * order_sign
    side_score_1 = lateral_coord[:, 1] * (-order_sign)
    side_score = torch.stack((side_score_0, side_score_1), dim=1)
    side_error = torch.clamp(side_margin - side_score, min=0.0, max=max_crossing_error)

    penalty = order_error.square() + 0.5 * side_error.square().mean(dim=1)
    return _safe_nonnegative_reward_bounded(penalty, max_crossing_error * max_crossing_error * 1.5)


@torch.jit.script
def _feet_parallel_contact_penalty_core(
    foot_quat_w: torch.Tensor,
    contact_force_w: torch.Tensor,
    sole_normal_axis: torch.Tensor,
    ground_normal_w: torch.Tensor,
    contact_threshold: float,
) -> torch.Tensor:
    foot_quat_w = foot_quat_w / torch.norm(foot_quat_w, dim=-1, keepdim=True).clamp(min=1e-8)
    foot_rot_w = quat_to_rot_matrix_wxyz(foot_quat_w)

    local_axis = _normalize_vec3_tensor(sole_normal_axis)
    sole_normal_w = torch.einsum("nbij,j->nbi", foot_rot_w, local_axis)
    sole_normal_w = torch.nan_to_num(sole_normal_w, nan=0.0, posinf=0.0, neginf=0.0)

    ground_normal = _normalize_vec3_tensor(ground_normal_w)
    cos_angle = torch.sum(sole_normal_w * ground_normal, dim=-1).abs()
    cos_angle = torch.nan_to_num(cos_angle, nan=0.0, posinf=1.0, neginf=0.0)
    cos_angle = torch.clamp(cos_angle, 0.0, 1.0)

    angle_error = torch.acos(cos_angle)

    contact_norm = torch.norm(contact_force_w, dim=-1)
    contact_norm = torch.nan_to_num(contact_norm, nan=0.0, posinf=0.0, neginf=0.0)
    in_contact_f = (contact_norm > contact_threshold).to(dtype=angle_error.dtype)

    penalty_per_foot = angle_error.square() * in_contact_f
    num_contacts = in_contact_f.sum(dim=1).clamp(min=1.0)
    penalty = penalty_per_foot.sum(dim=1) / num_contacts
    return _safe_nonnegative_reward_bounded(penalty, 2.4674011002723395)


@torch.jit.script
def _feet_symmetry_penalty_core(
    pos_b: torch.Tensor,
    forward_dir: torch.Tensor,
    foot0_ahead_avg: torch.Tensor,
    foot1_ahead_avg: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_dir = _normalize_vec3_tensor(forward_dir)
    forward_coord = _project_along(pos_b, forward_dir)
    foot0_ahead = (forward_coord[:, 0] > forward_coord[:, 1]).to(dtype=forward_coord.dtype)
    foot1_ahead = 1.0 - foot0_ahead

    next_foot0_ahead_avg = (1.0 - alpha) * foot0_ahead_avg + alpha * foot0_ahead
    next_foot1_ahead_avg = (1.0 - alpha) * foot1_ahead_avg + alpha * foot1_ahead

    diff = next_foot0_ahead_avg - next_foot1_ahead_avg
    penalty = diff.square()
    return _safe_nonnegative_reward_bounded(penalty, 1.0), next_foot0_ahead_avg, next_foot1_ahead_avg, forward_coord


@torch.jit.script
def _track_base_height_exp_core(base_z: torch.Tensor, target_height: float, std: float) -> torch.Tensor:
    height_error = base_z - target_height
    return torch.exp(-height_error.square() / (std * std))


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
              With current settings this means foot 0 should have smaller +Y than foot 1.
        +1.0: foot 0 should be on the positive side of foot 1 along ``lateral_dir_b``.

    For your debug logs, crossed poses had foot_0_lateral > foot_1_lateral, so the
    correct setting is the default: ``expected_foot0_lateral_order = -1.0``.
    """
    pos_b, _ = get_feet_pose_base(env, asset_cfg)
    lateral_dir = _unit_vec(lateral_dir_b, env.device, dtype=pos_b.dtype)

    penalty = _feet_crossing_penalty_core(
        pos_b,
        lateral_dir,
        float(min_lateral_separation),
        float(expected_foot0_lateral_order),
        float(side_margin),
        float(max_crossing_error),
    )

    # Debug values are intentionally kept outside the scripted core.
    lateral_coord = _project_along(pos_b, lateral_dir)
    order_sign = 1.0 if expected_foot0_lateral_order >= 0.0 else -1.0
    delta_01 = lateral_coord[:, 0] - lateral_coord[:, 1]
    signed_separation = order_sign * delta_01
    order_error = torch.clamp(min_lateral_separation - signed_separation, min=0.0, max=max_crossing_error)
    side_score = torch.stack((lateral_coord[:, 0] * order_sign, lateral_coord[:, 1] * (-order_sign)), dim=1)
    side_error = torch.clamp(side_margin - side_score, min=0.0, max=max_crossing_error)

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
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]

    local_axis = _unit_vec(sole_normal_axis, env.device, dtype=foot_quat_w.dtype)
    ground_normal = _unit_vec(ground_normal_w, env.device, dtype=foot_quat_w.dtype)

    penalty = _feet_parallel_contact_penalty_core(
        foot_quat_w,
        contact_force,
        local_axis,
        ground_normal,
        float(contact_threshold),
    )

    # Debug values are intentionally kept outside the scripted core.
    foot_quat_w_dbg = foot_quat_w / torch.norm(foot_quat_w, dim=-1, keepdim=True).clamp(min=1e-8)
    foot_rot_w = quat_to_rot_matrix_wxyz(foot_quat_w_dbg)
    sole_normal_w = torch.einsum("nbij,j->nbi", foot_rot_w, local_axis)
    sole_normal_w = torch.nan_to_num(sole_normal_w, nan=0.0, posinf=0.0, neginf=0.0)
    cos_angle = torch.sum(sole_normal_w * ground_normal, dim=-1).abs()
    cos_angle = torch.nan_to_num(cos_angle, nan=0.0, posinf=1.0, neginf=0.0)
    cos_angle = torch.clamp(cos_angle, 0.0, 1.0)
    angle_error = torch.acos(cos_angle)
    contact_norm = torch.norm(contact_force, dim=-1)
    contact_norm = torch.nan_to_num(contact_norm, nan=0.0, posinf=0.0, neginf=0.0)
    in_contact = contact_norm > contact_threshold

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
    alpha: float = 0.001,
    forward_dir_b: tuple[float, float, float] = FEET_FORWARD_DIR_B,
    debug: bool = False,
    debug_every: int = 100,
    debug_env_id: int = 0,
) -> torch.Tensor:
    """Penalize one foot being ahead much more often than the other.

    The term keeps an exponential moving average per environment of which foot
    index is ahead along the robot forward axis. A symmetric alternating gait
    drives both averages toward 0.5, while a static one-foot-ahead posture drives
    one average toward 1.0 and the other toward 0.0.
    """
    pos_b, _ = get_feet_pose_base(env, asset_cfg)
    forward_dir = _unit_vec(forward_dir_b, env.device, dtype=pos_b.dtype)

    num_envs = pos_b.shape[0]
    if (
        not hasattr(env, "_feet_ahead_avg")
        or env._feet_ahead_avg["foot0"].shape[0] != num_envs
        or env._feet_ahead_avg["foot0"].device != pos_b.device
    ):
        env._feet_ahead_avg = {
            "foot0": torch.zeros(num_envs, device=pos_b.device, dtype=pos_b.dtype),
            "foot1": torch.zeros(num_envs, device=pos_b.device, dtype=pos_b.dtype),
        }

    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        env._feet_ahead_avg["foot0"] = torch.where(
            reset_mask,
            torch.zeros_like(env._feet_ahead_avg["foot0"]),
            env._feet_ahead_avg["foot0"],
        )
        env._feet_ahead_avg["foot1"] = torch.where(
            reset_mask,
            torch.zeros_like(env._feet_ahead_avg["foot1"]),
            env._feet_ahead_avg["foot1"],
        )

    penalty, foot0_avg, foot1_avg, forward_coord = _feet_symmetry_penalty_core(
        pos_b,
        forward_dir,
        env._feet_ahead_avg["foot0"],
        env._feet_ahead_avg["foot1"],
        float(alpha),
    )
    env._feet_ahead_avg["foot0"] = foot0_avg.detach()
    env._feet_ahead_avg["foot1"] = foot1_avg.detach()

    # Debug values are intentionally kept outside the scripted core.
    foot0_ahead = forward_coord[:, 0] > forward_coord[:, 1]
    diff = foot0_avg - foot1_avg

    if debug and hasattr(env, "common_step_counter") and debug_every > 0:
        if env.common_step_counter % debug_every == 0:
            i = _debug_index(env, debug_env_id)
            print("\n[feet_symmetry_penalty]")
            print(f"  step: {env.common_step_counter}")
            print(f"  forward_dir_b: {forward_dir_b}")
            print(f"  alpha: {alpha:.6f}")
            print(f"  foot_0_pos_b: {pos_b[i, 0].detach().cpu().numpy()}")
            print(f"  foot_1_pos_b: {pos_b[i, 1].detach().cpu().numpy()}")
            print(f"  foot_0_forward: {forward_coord[i, 0].item():.4f}")
            print(f"  foot_1_forward: {forward_coord[i, 1].item():.4f}")
            print(f"  foot_0_ahead_now: {bool(foot0_ahead[i].item())}")
            print(f"  foot_0_ahead_avg: {foot0_avg[i].item():.4f}")
            print(f"  foot_1_ahead_avg: {foot1_avg[i].item():.4f}")
            print(f"  ahead_avg_diff: {diff[i].item():.4f}")
            print(f"  penalty_unweighted: {penalty[i].item():.6f}")

    return penalty


@torch.jit.script
def base_too_low_core(body_pos_w: torch.Tensor, minimum_height: float) -> torch.Tensor:
    return body_pos_w[:, 0, 2] < minimum_height


def terminate_if_base_too_low(env, minimum_height: float = 0.8):
    return base_too_low_core(env.scene["robot"].data.body_pos_w, minimum_height)


@torch.jit.script
def track_base_height_exp_core(root_pos_w: torch.Tensor, target_height: float, std: float) -> torch.Tensor:
    height_error = root_pos_w[:, 2] - target_height
    return torch.exp(-height_error.square() / (std * std))


def track_base_height_exp(env, target_height=1.4, std=0.2, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return track_base_height_exp_core(robot.data.root_pos_w, target_height, std)


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

    forward_vel_x = RewTerm(
        func=mdp.forward_vel_x_world,
        weight=REWARD_WEIGHTS["forward_vel_x"],
        params={"asset_cfg": SceneEntityCfg("robot")},
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
            "alpha": REWARD_PARAMS.gait_symmetry["alpha"],
            "forward_dir_b": FEET_FORWARD_DIR_B,
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
        update_interval=TRAINING_PARAMS.actions.tendon_update_interval,
        model_type=TRAINING_PARAMS.actions.tendon_model_type,
        identix_bundle_dir=TRAINING_PARAMS.actions.tendon_identix_bundle_dir,
        identix_repo_path=TRAINING_PARAMS.actions.tendon_identix_repo_path,
        identix_compile=TRAINING_PARAMS.actions.tendon_identix_compile,
        identix_transfer=TRAINING_PARAMS.actions.tendon_identix_transfer,
        identix_force_scale=TRAINING_PARAMS.actions.tendon_identix_force_scale,
        identix_force_sign=TRAINING_PARAMS.actions.tendon_identix_force_sign,
        identix_apply_mode=TRAINING_PARAMS.actions.tendon_identix_apply_mode,
    )
