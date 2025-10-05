# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =========================
# Imports
# =========================
# [Modified] 改动：扁平环境不再从 ForrestRoughEnvCfg 继承，因此直接引入与 rough 环境相同的基础依赖
# [Modified] Change: Flat env no longer inherits ForrestRoughEnvCfg; import the same base deps as rough.
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# ★★★ 修改点：导入改为 ForrestLocomotionVelocityEnvCfg ★★★
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ForrestLocomotionVelocityEnvCfg,
    RewardsCfg,
)

from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets.articulation import Articulation

from isaaclab_assets import FORREST_CFG  # isort: skip

import torch

# ===========================================================
# === 🔹 新增部分：YAML 加载与覆盖工具 (Added)
# ===========================================================
from pathlib import Path
import os, yaml

def _apply_overrides(obj, overrides: dict):
    """
    递归地将 YAML 参数写入配置对象 (Recursively apply dict overrides to dataclass/config objects).
    - 现在支持 obj 为 dict 的情况
    - 增加了 terminations 的便捷路由
    """
    # === 支持 obj 是 dict 的情况（字典级递归）===
    if isinstance(obj, dict):
        for k, v in overrides.items():
            if isinstance(v, dict):
                if k in obj:
                    # 可能是 dict，也可能是配置对象（如 SceneEntityCfg）
                    _apply_overrides(obj[k], v)
                else:
                    obj[k] = v
            else:
                # 小抛光：把列表转成 tuple，避免下游类型假设不一致
                if isinstance(v, list) and k in ("body_names", "joint_names"):
                    v = tuple(v)
                obj[k] = v
        return

    for k, v in overrides.items():
        # === NEW/UPDATED: terminations 容器到具体 term 的路由（例如 base_contact）===
        if k == "base_contact" and hasattr(obj, "base_contact") and isinstance(v, dict):
            _apply_overrides(getattr(obj, "base_contact"), v)
            continue

        if not hasattr(obj, k):
            # 特殊键映射
            if k == "robot_prim_path" and hasattr(obj, "robot"):
                obj.robot = obj.robot.replace(prim_path=v)
                continue
            if k == "height_scanner_prim_path" and hasattr(obj, "height_scanner"):
                obj.height_scanner.prim_path = v
                continue
            if k == "asset_body_names" and hasattr(obj, "params"):
                if "asset_cfg" in obj.params:
                    # 小抛光：list -> tuple
                    obj.params["asset_cfg"].body_names = tuple(v) if isinstance(v, list) else v
                continue
            if k == "base_contact_body_names":
                # 允许从 terminations 层级快捷配置 body_names
                if hasattr(obj, "base_contact") and hasattr(obj.base_contact, "params"):
                    bc = obj.base_contact
                    if "sensor_cfg" in bc.params:
                        bc.params["sensor_cfg"].body_names = tuple(v)
                        continue

            if k in ("pose_range", "velocity_range", "position_range") and hasattr(obj, "params"):
                val = v
                if isinstance(val, list):
                    val = tuple(val)  # 与内部 tuple 约定保持一致
                obj.params[k] = val
                continue


            continue

        cur = getattr(obj, k)
        if isinstance(v, dict) and cur is not None:
            _apply_overrides(cur, v)
        else:
            # 小抛光：把列表转成 tuple，避免下游类型假设不一致
            if isinstance(v, list) and k in ("body_names", "joint_names"):
                v = tuple(v)
            setattr(obj, k, v)

# def _load_yaml_here(file_name: str) -> dict:
# # 作用：在当前 Python 文件同目录下找 file_name，读 YAML，返回 dict（文件不存在就返回空 dict）。
#     path = Path(__file__).with_name(file_name)
#     if not path.exists():
#         return {}
#     with open(path, "r") as f:
#         return yaml.safe_load(f) or {}

def _load_yaml_here(file_name: str) -> dict:
    here = Path(__file__).resolve().parent
    candidates = [
        here / file_name,  # 运行时同目录（extscache/site-packages）
        Path(os.getenv("ISAACLAB_OVERRIDES_DIR", "")) / file_name if os.getenv("ISAACLAB_OVERRIDES_DIR") else None,
        Path.cwd() / file_name,  # 当前工作目录
    ]
    candidates = [p for p in candidates if p is not None]

    print(f"[OVERRIDE] __file__ = {Path(__file__).resolve()}")
    print(f"[OVERRIDE] search: {', '.join(str(p) for p in candidates)}")
    for p in candidates:
        if p.exists():
            print(f"[OVERRIDE] using: {p}")
            with open(p, "r") as f:
                data = yaml.safe_load(f) or {}
            print(f"[OVERRIDE] loaded keys: {list(data.keys())}")
            return data
    print("[OVERRIDE] no overrides found.")
    return {}


# === NEW/UPDATED ===
def _apply_reward_overrides(rewards_obj, rdict: dict):
    """
    通用奖励覆盖：若 term 存在，则设置 weight / 常见 params 字段。
    这样 YAML 中的大多数 reward 键都能直接生效（避免逐项硬编码）。
    """
    for term_name, term_cfg in rdict.items():
        if not hasattr(rewards_obj, term_name):
            continue
        term = getattr(rewards_obj, term_name)
        # 标量：直接当作 weight
        if isinstance(term_cfg, (int, float)):
            term.weight = term_cfg
            continue
        if not isinstance(term_cfg, dict):
            continue
        # weight
        if "weight" in term_cfg:
            term.weight = term_cfg["weight"]
        # 常见 params
        if hasattr(term, "params"):
            if "std" in term_cfg:
                term.params["std"] = term_cfg["std"]
            if "threshold" in term_cfg:
                term.params["threshold"] = term_cfg["threshold"]
            if "alpha" in term_cfg:
                term.params["alpha"] = term_cfg["alpha"]
            if "joint_names" in term_cfg and "asset_cfg" in term.params:
                jn = term_cfg["joint_names"]
                term.params["asset_cfg"].joint_names = tuple(jn) if isinstance(jn, list) else jn

# =========================
# Utility / helpers (ported from rough_env_cfg)
# =========================
FEET_CFG = SceneEntityCfg(
    "robot",
    body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
)

def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (wxyz) to rotation matrices."""
    # q: [N, 4] in (x, y, z, w) order (check Isaac Lab ordering!)
    x, y, z, w = q.unbind(-1)
    # rotation matrix components
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    rot = torch.stack([
        1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy),
    ], dim=-1)
    rot = rot.view(-1, 3, 3)
    return rot


def get_feet_pose_base(env, feet_cfg: SceneEntityCfg = FEET_CFG):
    robot = env.scene[feet_cfg.name]
    ids = feet_cfg.body_ids  # get the ids of the feet

    # in the world frame
    pos_w = robot.data.body_pos_w[:, ids, :]      # [N, 2, 3] position of the 2 feet in the world frame
    quat_w = robot.data.body_quat_w[:, ids, :]    # [N, 2, 4] Quaternion of the 2 feet in the world frame

    base_pos = robot.data.root_pos_w[:, None, :]  # [N, 1, 3]
    base_quat = robot.data.root_quat_w            # [N, 4]

    # relative in world frame
    rel = pos_w - base_pos                        # [N, 2, 3]

    # rotate into base frame
    R = quat_to_rot_matrix(base_quat)             # [N, 3, 3]
    rel_b = torch.einsum("nij,nkj->nki", R.transpose(1, 2), rel)

    # for orientations, you can skip for now if you only need positions
    return rel_b, quat_w


def feet_symmetry_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
    ),
    alpha: float = 0.01,
) -> torch.Tensor:
    # initialize EMA if not already present
    # EMA: Exponential Moving Average
    if not hasattr(env, "_feet_avg"):
        env._feet_avg = {
            "left": torch.zeros(env.num_envs, device=env.device),
            "right": torch.zeros(env.num_envs, device=env.device),
        }

    # get base-frame foot positions
    pos_b, _ = get_feet_pose_base(env, asset_cfg) # this is in the robotic body frame
    x = pos_b[:, :, 0]

    # indicators
    left_ahead = (x[:, 0] > x[:, 1]).float()
    right_ahead = 1.0 - left_ahead

    # EMA update
    env._feet_avg["left"] = (1 - alpha) * env._feet_avg["left"] + alpha * left_ahead
    env._feet_avg["right"] = (1 - alpha) * env._feet_avg["right"] + alpha * right_ahead

    # symmetry difference
    diff = env._feet_avg["left"] - env._feet_avg["right"]

    # L2 penalty
    penalty = diff.pow(2)

    return penalty


def terminate_if_base_too_low(env, minimum_height: float = 0.8):
    # Torch tensor: (num_envs, num_bodies, 3)
    body_pos = env.scene["robot"].data.body_pos_w
    # z-coordinate of base body (index 0 or use name lookup)
    base_z = body_pos[:, 0, 2]  # shape (num_envs,)
    # return a torch.BoolTensor mask
    return base_z < minimum_height


# =========================
# Rewards (ported & aligned with rough_env_cfg)
# =========================
@configclass
class ForrestFlatRewards(RewardsCfg):
    #   Reward terms for the MDP.
    #   positive weight: reward
    #   negative weight: penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Roll: rotation around the x-axis
    # Pitch: rotation around the y-axis
    # Yaw: rotation around the z-axis
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,  # Yaw: rotation around the z-axis
        weight=3.0, # 1.0
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,  # flat env keeps explicit emphasis on yaw tracking
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
            ),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
            ),
        },
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l1_acetabulofemoral_lateral",
                    "l5_metatarsophalangeal",
                    "l6_interphalangeal",
                    "r1_acetabulofemoral_lateral",
                    "r5_metatarsophalangeal",
                    "r6_interphalangeal",
                ],
            )
        },
    )

    hip_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
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

    # The robot’s forward direction is the x-axis.
    # Roll: rotation around the x-axis
    # Pitch: rotation around the y-axis
    # Yaw: rotation around the z-axis
    gait_symetry = RewTerm(
        func=feet_symmetry_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
            ),
            "alpha": 0.001,
        },
    )


# =========================
# Flat Env Config (now independent of rough)
# =========================
@configclass
class ForrestFlatEnvCfg(ForrestLocomotionVelocityEnvCfg):
    # [Modified] 改动：不再继承 ForrestRoughEnvCfg，而是直接继承 LocomotionVelocityRoughEnvCfg
    # [Modified] Change: inherit directly from LocomotionVelocityRoughEnvCfg instead of ForrestRoughEnvCfg
    rewards: ForrestFlatRewards = ForrestFlatRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # [Modified] 改动：为平地环境显式设置机器人、传感器等，与 rough 中保持一致的基础配置
        # [Modified] Change: explicitly set robot & sensors similar to rough baseline for consistency.
        self.scene.robot = FORREST_CFG.replace(prim_path="{ENV_REGEX_NS}/Forrest_URDF")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Forrest_URDF/base_link"

        # Solve issue with dropping contacts (keep same robustness settings)
        self.sim.physx.gpu_collision_stack_size = 160 * 1024 * 1024  # 80 MB

        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Forrest_URDF/(S45_Digit_Assyv2_1|S45_Digit_Assyv2_mirror_1|base_link|Differential_Cage_Assyv7_mirror_1|Differential_Cage_Assyv7_1|Knee_Assyv9_mirror_1|Knee_Assyv9_1)",
            update_period=0.0,  # update every sim step
            history_length=6,
            debug_vis=True,
            track_air_time=True,
        )

        # Terrain: force flat plane and disable generators/curriculum
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        # no height scan (keep components available as in original comment)
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None

        # Randomization / resets (align with rough defaults)
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com.params["asset_cfg"].body_names = ["base_link"]

        # Rewards: keep upright, discourage twitchy actions; flat keeps yaw tracking weight emphasized
        # self.rewards.lin_vel_z_l2.weight = 0.0  # disables vertical velocity penalty
        self.rewards.undesired_contacts = None  # removes undesired contacts penalty
        self.rewards.flat_orientation_l2.weight = -1.0  # keeps base upright
        self.rewards.action_rate_l2.weight = -0.0025  # penalizes fast changes in actions

        # DOF accelerations penalty
        self.rewards.dof_acc_l2.weight = -1.3e-7 # -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "l0_acetabulofemoral_roll",
                "l1_acetabulofemoral_lateral",
                "l2_pseudo_acetabulofemoral_flexion",
                "l3f_femorotibial_front",
                "l5_metatarsophalangeal",
                "l6_interphalangeal",
                "r0_acetabulofemoral_roll",
                "r1_acetabulofemoral_lateral",
                "r2_pseudo_acetabulofemoral_flexion",
                "r3f_femorotibial_front",
                "r5_metatarsophalangeal",
                "r6_interphalangeal",
            ],
        )

        # DOF torques penalty (disabled like rough)
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "l0_acetabulofemoral_roll",
                "l1_acetabulofemoral_lateral",
                "l2_pseudo_acetabulofemoral_flexion",
                "l3f_femorotibial_front",
                "l5_metatarsophalangeal",
                "l6_interphalangeal",
                "r0_acetabulofemoral_roll",
                "r1_acetabulofemoral_lateral",
                "r2_pseudo_acetabulofemoral_flexion",
                "r3f_femorotibial_front",
                "r5_metatarsophalangeal",
                "r6_interphalangeal",
            ],
        )

        # Commands (keep symmetry with rough; specific play-time ranges in PLAY)
        self.commands.base_velocity.ranges.lin_vel_x = (-4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations (reuse the safe baseline)
        self.terminations.base_contact.params["sensor_cfg"].body_names = (
            "base_link",
            "Differential_Cage_Assyv7_mirror_1",
            "Differential_Cage_Assyv7_1",
            "Knee_Assyv9_mirror_1",
            "Knee_Assyv9_1",
        )
        # Example of adding custom termination (not enabled by default):
        # self.terminations.base_too_low = TerminationTermCfg(func=terminate_if_base_too_low, params={"minimum_height": 0.8})

        # ===========================================================
        # === 🔹 新增部分：从 YAML 文件 flat_env_overrides.yaml 读取并覆盖参数
        # ===========================================================
        _ov = _load_yaml_here("flat_env_overrides.yaml")
        if _ov:
            if "scene" in _ov: _apply_overrides(self.scene, _ov["scene"])
            if "sim" in _ov: _apply_overrides(self.sim, _ov["sim"])
            for key in ("curriculum", "events", "commands", "terminations"):
                if key in _ov and hasattr(self, key):
                    _apply_overrides(getattr(self, key), _ov[key])
            if "rewards" in _ov:
                r = _ov["rewards"]

                # === 支持 YAML 用 null 禁用某个 reward ===
                for term_name, term_cfg in list(r.items()):
                    if term_cfg is None and hasattr(self.rewards, term_name):
                        setattr(self.rewards, term_name, None)
                        del r[term_name]  # 从 r 里移除，避免后面再处理这个 term

                if "flat_orientation_l2" in r:
                    self.rewards.flat_orientation_l2.weight = r["flat_orientation_l2"]
                if "action_rate_l2" in r:
                    self.rewards.action_rate_l2.weight = r["action_rate_l2"]
                if "lin_vel_z_l2" in r and hasattr(self.rewards, "lin_vel_z_l2"):
                    self.rewards.lin_vel_z_l2.weight = r["lin_vel_z_l2"]
                if "dof_acc_l2" in r:
                    self.rewards.dof_acc_l2.weight = r["dof_acc_l2"]["weight"]
                    self.rewards.dof_acc_l2.params["asset_cfg"].joint_names = r["dof_acc_l2"]["joint_names"]
                if "dof_torques_l2" in r:
                    self.rewards.dof_torques_l2.weight = r["dof_torques_l2"]["weight"]
                    self.rewards.dof_torques_l2.params["asset_cfg"].joint_names = r["dof_torques_l2"]["joint_names"]
                if "gait_symmetry" in r:
                    self.rewards.gait_symetry.weight = r["gait_symmetry"]["weight"]
                    self.rewards.gait_symetry.params["alpha"] = r["gait_symmetry"]["alpha"]

                # === 通用奖励兜底覆盖（让 YAML 里更多 reward 键直接生效）===
                _apply_reward_overrides(self.rewards, r)



# ===========================================================
# PLAY variant
# ===========================================================

class ForrestFlatEnvCfg_PLAY(ForrestFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # Command ranges tailored for play mode on flat terrain
        self.commands.base_velocity.ranges.lin_vel_x = (-6.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # === ✅ 新增：允许 YAML 覆盖 PLAY 模式参数 ===
        _ov = _load_yaml_here("flat_env_overrides.yaml")
        if _ov and "play" in _ov:
            play_cfg = _ov["play"]

            # 可选地让 randomization_off 真正“可控”
            if "randomization_off" in play_cfg:
                roff = bool(play_cfg["randomization_off"])
                self.observations.policy.enable_corruption = not roff
                if roff:
                    self.events.base_external_force_torque = None
                    self.events.push_robot = None

            # 允许通过 YAML 覆盖 commands / terminations / events 等
            _apply_overrides(self, play_cfg)

# [Added] 加在文件最后，确保导入 flat_env_cfg.py 时能打印提示
# [Added] Added at the very end of the file, ensures message prints when flat_env_cfg.py is imported
print("\n================= 2025.10.5 ForrestFlatEnvCfg 已加载 (Flat Environment Config Loaded) =================\n")