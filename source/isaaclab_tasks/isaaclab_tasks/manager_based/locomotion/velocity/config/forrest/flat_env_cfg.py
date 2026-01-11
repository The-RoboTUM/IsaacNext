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


        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.termination_penalty.weight = -200.0
        
        self.rewards.track_lin_vel_xy_exp.weight = 10.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
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


        # disable randomization for play
        self.observations.policy.enable_corruption = False
