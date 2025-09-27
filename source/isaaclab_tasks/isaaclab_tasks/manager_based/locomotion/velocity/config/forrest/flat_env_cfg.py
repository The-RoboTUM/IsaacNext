# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from .rough_env_cfg import ForrestRoughEnvCfg

def terminate_if_base_too_low(env, minimum_height: float = 0.8):
    # Torch tensor: (num_envs, num_bodies, 3)
    body_pos = env.scene["robot"].data.body_pos_w

    # z-coordinate of base body (index 0 or use name lookup)
    base_z = body_pos[:, 0, 2]  # shape (num_envs,)

    # return a torch.BoolTensor mask
    return base_z < minimum_height

@configclass
class ForrestFlatEnvCfg(ForrestRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None


        # Rewards
        # self.rewards.lin_vel_z_l2.weight = 0.0  # disables vertical velocity penalty

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 0.75
        # self.rewards.feet_air_time.weight = 0.0
        # self.rewards.feet_slide.weight = 0.0
        # self.rewards.dof_torques_l2.weight = -2.0e-6

        # # Commands
        # self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.lin_vel_z = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_y = (-1.0, 1.0)

        # Terminations
        # self.terminations.base_too_low = TerminationTermCfg(
        #     func=terminate_if_base_too_low,
        #     params={"minimum_height": 1.0},
        # )


class ForrestFlatEnvCfg_PLAY(ForrestFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.scene.num_envs = 50
        # self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
