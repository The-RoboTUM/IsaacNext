from dataclasses import MISSING


import isaaclab.utils.string as string_utils
import isaaclab.envs
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm, ActionTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from isaaclab.sensors import ContactSensorCfg
##
# Pre-defined configs
##
from isaaclab_assets import FORREST_CFG  # isort: skip

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from isaaclab.assets.articulation import Articulation

FEET_CFG = SceneEntityCfg(
    "robot",
    body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
)

import torch
# from isaaclab.utils import torch as torch_utils

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

@configclass
class ForrestRewards(RewardsCfg):
 #   Reward terms for the MDP.

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=("S45_Digit_Assyv2_1",
                                                                       "S45_Digit_Assyv2_mirror_1")),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=("S45_Digit_Assyv2_1",
                                                                       "S45_Digit_Assyv2_mirror_1")),
            "asset_cfg": SceneEntityCfg("robot", body_names=("S45_Digit_Assyv2_1",
                                                             "S45_Digit_Assyv2_mirror_1")),
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


    gait_symetry = RewTerm(
        func=feet_symmetry_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=("S45_Digit_Assyv2_1", "S45_Digit_Assyv2_mirror_1"),
            ),
            "alpha":0.001,
        },
    )



# class TendonActionTerm(ActionTerm):
#     """Joint Action term that applies the processed actions to the articulation's joints as position commands.
#     Incorporates Tendon Math"""
#
#     r"""Base class for joint actions.
#
#         This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
#         These transformations can be configured to be applied to a subset of the articulation's joints.
#
#         Mathematically, the action term is defined as:
#
#         .. math::
#
#            \text{action} = \text{offset} + \text{scaling} \times \text{input action}
#
#         where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
#         is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
#         action, and :math:`\text{input action}` is the input action from the user.
#
#         Based on above, this kind of action transformation ensures that the input and output actions are in the same
#         units and dimensions. The child classes of this action term can then map the output action to a specific
#         desired command of the articulation's joints (e.g. position, velocity, etc.).
#         """
#
#
#     """The configuration of the action term."""
#     cfg : TendonActionTermCfg
#     """The configuration of the action term."""
#     _asset: Articulation
#     """The articulation asset on which the action term is applied."""
#     _scale: torch.Tensor | float
#     """The scaling factor applied to the input action."""
#     _offset: torch.Tensor | float
#     """The offset applied to the input action."""
#     _clip: torch.Tensor
#     """The clip applied to the input action."""
#
#     def __init__(self, cfg: TendonActionTermCfg, env: ManagerBasedEnv):
#         # initialize the action term
#         super().__init__(cfg, env)
#
#         # resolve the joints over which the action term is applied
#         self._joint_ids_all, self._joint_names_all = self._asset.find_joints(self.__joint_names_actuated + self.__joint_names_tendon_driven, preserve_order=True)
#         self._joint_ids_actuated, self._joint_names_actuated = self._asset.find_joints(self.__joint_names_actuated, preserve_order=True)
#         self._joint_ids_tendon_driven, self._joint_names_tendon_driven = self._asset.find_joints(self.__joint_names_tendon_driven, preserve_order=True)
#
#         self._num_joints_all = len(self._joint_ids_all)
#         self._num_actuated_joints = len(self._joint_ids_actuated)
#         # log the resolved joint names for debugging
#         omni.log.info(
#             f"Resolved joint names for the action term {self.__class__.__name__}:"
#             f" {self._joint_names_all} [{self._joint_ids_all}]"
#         )
#
#         # create tensors for raw and processed actions
#         self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
#         self._processed_direct_actions = torch.zeros_like(self.raw_actions)
#         self._processed_actions = torch.zeros(self.num_envs, self._num_joints_all, device=self.device)
#
#         # parse scale
#         if isinstance(cfg.scale, (float, int)):
#             self._scale = float(cfg.scale)
#         elif isinstance(cfg.scale, dict):
#             self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
#             # resolve the dictionary config
#             index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
#             self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
#         else:
#             raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
#
#         # parse clip
#         if self.cfg.clip is not None:
#             if isinstance(cfg.clip, dict):
#                 self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
#                     self.num_envs, self.action_dim, 1
#                 )
#                 index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
#                 self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
#             else:
#                 raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")
#
#         # use default joint positions as offset
#         self._offset = self._asset.data.default_joint_pos[:, self._joint_ids_actuated].clone()
#
#     """
#     Operations.
#     """
#     def apply_actions(self):
#         # set position targets
#         self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids_all)
#
#     def process_actions(self, actions: torch.Tensor):
#         # store the raw actions
#         self._raw_actions[:] = actions
#         # apply the affine transformations
#         self._processed_direct_actions = self._raw_actions * self._scale + self._offset
#         # clip actions
#         if self.cfg.clip is not None:
#             self._processed_direct_actions = torch.clamp(
#                 self._processed_direct_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
#             )
#         # step 1: transfer the directly controlled actions: use index of joint_idx_actuated in joint_idx_all
#         # step 2: for each tendon name, index of joint_idx in joint_idx_all: compute
#         # _processed_actions[0] -applied-to> joint_ids_all[0]
#
#     def reset(self, env_ids: Sequence[int] | None = None) -> None:
#         self._raw_actions[env_ids] = 0.0
#
#     """
#     Properties.
#     """
#
#     @property
#     def action_dim(self) -> int:
#         return self._num_actuated_joints
#
#     @property
#     def raw_actions(self) -> torch.Tensor:
#         return self._raw_actions
#
#     @property
#     def processed_actions(self) -> torch.Tensor:
#         return self._processed_actions
#
#     @property
#     def IO_descriptor(self) -> GenericActionIODescriptor:
#         """The IO descriptor of the action term.
#
#         This descriptor is used to describe the action term of the joint action.
#         It adds the following information to the base descriptor:
#         - joint_names: The names of the joints.
#         - scale: The scale of the action term.
#         - offset: The offset of the action term.
#         - clip: The clip of the action term.
#
#         Returns:
#             The IO descriptor of the action term.
#         """
#         super().IO_descriptor
#         self._IO_descriptor.shape = (self.action_dim,)
#         self._IO_descriptor.dtype = str(self.raw_actions.dtype)
#         self._IO_descriptor.action_type = "TendonActionTerm"
#         self._IO_descriptor.joint_names = self._joint_names_actuated
#         self._IO_descriptor.scale = self._scale
#         # FIXME: This is not correct. Add list support.
#         if self.cfg.clip is not None:
#             if isinstance(self._clip, torch.Tensor):
#                 self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
#             else:
#                 self._IO_descriptor.clip = self._clip
#         else:
#             self._IO_descriptor.clip = None
#         return self._IO_descriptor
#
#
#
#
#
# @configclass
# class TendonActionTermCfg:
#     class_type: type[ActionTerm] = TendonActionTerm
#     asset_name: str = MISSING
#     scale: float = 1.0
#
# @configclass
# class CustomActionManagerCfg:
#     joint_pos = TendonActionTermCfg()

@configclass
class ForrestRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ForrestRewards = ForrestRewards()
    # TODO: Implement Tendon Math
    # actions: CustomActionManagerCfg = CustomActionManagerCfg(joint_pos=TendonActionTermCfg(asset_name="robot", scale=0.5, use_default_offset=True))

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = FORREST_CFG.replace(prim_path="{ENV_REGEX_NS}/Forrest_URDF")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Forrest_URDF/base_link"

        # TEMP (Used only to make flat env model work)
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # self.curriculum.terrain_levels = None

        # Solve issue with dropping contacts
        self.sim.physx.gpu_collision_stack_size = 160 * 1024 * 1024  # 80 MB
        # self.sim.physx.gpu_max_rigid_patch_count = 400000

        # Sensors
        # self.scene.base_contact = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Forrest_URDF/Base_Assy_V2v18_1",
        #     update_period=0.0,
        #     history_length=1,
        #     debug_vis=True,
        #     track_air_time=True,
        # )
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Forrest_URDF/(S45_Digit_Assyv2_1|S45_Digit_Assyv2_mirror_1|base_link|Differential_Cage_Assyv7_mirror_1|Differential_Cage_Assyv7_1|Knee_Assyv9_mirror_1|Knee_Assyv9_1)",
            update_period=0.0,  # update every sim step
            history_length=6,
            debug_vis=True,
            track_air_time=True,
        )

        # Randomization
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

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0  # disables vertical velocity penalty
        self.rewards.undesired_contacts = None  # removes undesired contacts penalty
        self.rewards.flat_orientation_l2.weight = -1.0  # keeps base upright
        # self.rewards.flat_orientation_l2.weight = -0.0  # keeps base upright
        # self.rewards.action_rate_l2.weight = -0.005  # penalizes fast changes in actions
        self.rewards.action_rate_l2.weight = -0.0025  # penalizes fast changes in actions

        # DOF accelerations penalty
        self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.dof_acc_l2.weight = 0.0
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

        # DOF torques penalty
        # self.rewards.dof_torques_l2.weight = -1.5e-7
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

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ("base_link",
                                                                          "Differential_Cage_Assyv7_mirror_1",
                                                                          "Differential_Cage_Assyv7_1",
                                                                          "Knee_Assyv9_mirror_1",
                                                                          "Knee_Assyv9_1"
                                                                          )

        # # DEBUG
        # self.observations.policy.enable_corruption = False
        # # remove random pushing
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None


@configclass
class ForrestRoughEnvCfg_PLAY(ForrestRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.scene.num_envs = 50
        # self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None



