# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments - Standard TRON
##

gym.register(
    id="Isaac-Velocity-Rough-Tron-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TronRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TronRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Tron-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TronFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TronFlatPPORunnerCfg",
    },
)

##
# Register LimX Pointfoot - Blind Flat only
##

gym.register(
    id="Isaac-Limx-PF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.limx_pointfoot_env_cfg:PFBlindFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.limx_rsl_rl_ppo_cfg:PF_TRON1AFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Limx-PF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.limx_pointfoot_env_cfg:PFBlindFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.limx_rsl_rl_ppo_cfg:PF_TRON1AFlatPPORunnerCfg",
    },
)
