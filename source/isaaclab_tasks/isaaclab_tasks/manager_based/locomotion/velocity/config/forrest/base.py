# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.curriculums.command_bins_rl import BinnedVelocityCommandCfg, make_binned_velocity_curriculum_term
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg

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
    finite_observation,
    reset_root_state_uniform_all_envs_on_startup,
    terminate_if_base_too_low,
)
from .self_collision import SelectiveSelfCollisionCfg, create_selective_self_collision_filter


def _disable_zero_weight_rewards(rewards_cfg, reward_weights: dict[str, float]) -> None:
    """Remove zero-weight Forrest reward terms so they are not evaluated each step."""
    for reward_name, weight in reward_weights.items():
        if float(weight) == 0.0 and hasattr(rewards_cfg, reward_name):
            setattr(rewards_cfg, reward_name, None)


def _forrest_ground_material_cfg() -> sim_utils.RigidBodyMaterialCfg:
    ground = FORREST_PARAMS.physics.ground
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=ground.static_friction,
        dynamic_friction=ground.dynamic_friction,
        restitution=ground.restitution,
        friction_combine_mode=ground.friction_combine_mode,
        restitution_combine_mode=ground.restitution_combine_mode,
    )


def _sanitize_policy_observations(policy_cfg) -> None:
    finite_modifier = ModifierCfg(func=finite_observation)
    for term_name, term_cfg in policy_cfg.__dict__.items():
        if term_cfg is None or not hasattr(term_cfg, "func"):
            continue
        modifiers = list(term_cfg.modifiers or [])
        if not any(getattr(modifier, "func", None) is finite_observation for modifier in modifiers):
            modifiers.append(finite_modifier)
        term_cfg.modifiers = modifiers


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
        self.scene.terrain.physics_material = _forrest_ground_material_cfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        if FORREST_PARAMS.robot.use_height_scanner:
            self.scene.height_scanner.prim_path = FORREST_PARAMS.robot.height_scanner_prim_path
        else:
            self.scene.height_scanner = None
            self.observations.policy.height_scan = None
        _sanitize_policy_observations(self.observations.policy)
        if (
            FORREST_PARAMS.physics.articulation.enabled_self_collisions
            and FORREST_PARAMS.physics.articulation.selective_self_collision_body_names
        ):
            self.scene.replicate_physics = False

        # Solve issue with dropping contacts
        self.sim.physx.gpu_collision_stack_size = FORREST_PARAMS.physics.physx_gpu_collision_stack_size
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = (
            FORREST_PARAMS.physics.physx_gpu_found_lost_aggregate_pairs_capacity
        )
        self.sim.physx.gpu_total_aggregate_pairs_capacity = (
            FORREST_PARAMS.physics.physx_gpu_total_aggregate_pairs_capacity
        )

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
        self.events.physics_material.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names=list(CONTACT_PARAMS.foot_body_names),
        )
        self.events.physics_material.params["static_friction_range"] = TRAINING_PARAMS.events.foot_static_friction_range
        self.events.physics_material.params["dynamic_friction_range"] = (
            TRAINING_PARAMS.events.foot_dynamic_friction_range
        )
        self.events.physics_material.params["restitution_range"] = TRAINING_PARAMS.events.foot_restitution_range
        self.events.physics_material.params["num_buckets"] = TRAINING_PARAMS.events.foot_material_num_buckets
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
        if (
            FORREST_PARAMS.physics.articulation.enabled_self_collisions
            and FORREST_PARAMS.physics.articulation.selective_self_collision_body_names
        ):
            self.events.filter_forrest_self_collisions = EventTerm(
                func=create_selective_self_collision_filter,
                mode="prestartup",
                params={
                    "cfg": SelectiveSelfCollisionCfg(
                        robot_path_template=FORREST_PARAMS.robot.prim_path.replace(
                            "{ENV_REGEX_NS}", "/World/envs/env_{env_id}"
                        ),
                        allowed_body_names=tuple(
                            FORREST_PARAMS.physics.articulation.selective_self_collision_body_names
                        ),
                        debug=FORREST_PARAMS.physics.articulation.selective_self_collision_debug,
                    )
                },
            )

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

        _disable_zero_weight_rewards(self.rewards, REWARD_WEIGHTS)

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
        if TRAINING_PARAMS.command_curriculum.enabled:
            self.commands.base_velocity = BinnedVelocityCommandCfg(
                asset_name="robot",
                resampling_time_range=(10.0, 10.0),
                rel_standing_envs=0.0,
                rel_heading_envs=0.0,
                heading_command=False,
                debug_vis=True,
                ranges=self.commands.base_velocity.ranges,
                curriculum=TRAINING_PARAMS.command_curriculum,
            )
            self.curriculum.command_bins = make_binned_velocity_curriculum_term(
                TRAINING_PARAMS.command_curriculum.command_name
            )
