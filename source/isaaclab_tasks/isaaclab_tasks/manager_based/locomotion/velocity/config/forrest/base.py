# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.forrest import get_forrest_cfg  # isort: skip

from .rl_env_cfg import (
    CONTACT_PARAMS,
    FORREST_PARAMS,
    REWARD_WEIGHTS,
    TRAINING_PARAMS,
    ForrestActionsCfg,
    ForrestRewards,
    _body_name_regex,
    actuated_joint_names,
    reset_root_state_uniform_all_envs_on_startup,
    terminate_if_base_too_low,
)


@configclass
class ForrestBaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Shared Forrest locomotion environment configuration.

    This class holds robot, sensors, common randomization, commands, rewards,
    and terminations. Terrain-specific variants live in ``active_envs.py``.
    """

    rewards: ForrestRewards = ForrestRewards()
    actions: ForrestActionsCfg = ForrestActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        if TRAINING_PARAMS.episode_length_s is not None:
            self.episode_length_s = TRAINING_PARAMS.episode_length_s

        # Scene
        self.scene.robot = get_forrest_cfg(FORREST_PARAMS).replace(prim_path=FORREST_PARAMS.robot.prim_path)
        self.scene.height_scanner.prim_path = FORREST_PARAMS.robot.height_scanner_prim_path

        # Solve issue with dropping contacts
        self.sim.physx.gpu_collision_stack_size = FORREST_PARAMS.physics.physx_gpu_collision_stack_size

        # Sensors
        self.scene.contact_forces = ContactSensorCfg(
            prim_path=f"{FORREST_PARAMS.robot.prim_path}/{_body_name_regex(CONTACT_PARAMS.contact_sensor_body_names)}",
            update_period=CONTACT_PARAMS.update_period,
            history_length=CONTACT_PARAMS.history_length,
            debug_vis=CONTACT_PARAMS.debug_vis,
            track_air_time=CONTACT_PARAMS.track_air_time,
        )

        # Randomization
        if TRAINING_PARAMS.events.disable_push_robot:
            self.events.push_robot = None
        if TRAINING_PARAMS.events.disable_add_base_mass:
            self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (
            TRAINING_PARAMS.events.reset_robot_joint_position_range
        )
        self.events.base_external_force_torque.params["asset_cfg"].body_names = list(
            TRAINING_PARAMS.events.external_force_body_names
        )
        self.events.reset_base.params = {
            "pose_range": TRAINING_PARAMS.events.reset_base_pose_range,
            "velocity_range": TRAINING_PARAMS.events.reset_base_velocity_range,
        }
        if TRAINING_PARAMS.events.randomize_initial_base_pose:
            self.events.startup_reset_base = EventTerm(
                func=reset_root_state_uniform_all_envs_on_startup,
                mode="startup",
                params={
                    "pose_range": TRAINING_PARAMS.events.reset_base_pose_range,
                    "velocity_range": TRAINING_PARAMS.events.reset_base_velocity_range,
                },
            )
        self.events.base_com.params["asset_cfg"].body_names = list(TRAINING_PARAMS.events.base_com_body_names)

        # Rewards
        self.rewards.lin_vel_z_l2.weight = REWARD_WEIGHTS["lin_vel_z_l2"]
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = REWARD_WEIGHTS["flat_orientation_l2"]
        self.rewards.action_rate_l2.weight = REWARD_WEIGHTS["action_rate_l2"]

        self.rewards.dof_acc_l2.weight = REWARD_WEIGHTS["dof_acc_l2"]
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "l0_acetabulofemoral_roll",
                "l1_acetabulofemoral_lateral",
                "l2_pseudo_acetabulofemoral_flexion",
                "r0_acetabulofemoral_roll",
                "r1_acetabulofemoral_lateral",
                "r2_pseudo_acetabulofemoral_flexion",
            ],
        )

        self.rewards.dof_torques_l2.weight = REWARD_WEIGHTS["dof_torques_l2"]
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=actuated_joint_names,
        )

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = CONTACT_PARAMS.base_termination_body_names
        self.terminations.base_too_low = TerminationTermCfg(
            func=terminate_if_base_too_low,
            params={"minimum_height": TRAINING_PARAMS.terminations.base_too_low_height},
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = TRAINING_PARAMS.commands.lin_vel_x
        self.commands.base_velocity.ranges.lin_vel_y = TRAINING_PARAMS.commands.lin_vel_y
        self.commands.base_velocity.ranges.ang_vel_z = TRAINING_PARAMS.commands.ang_vel_z
