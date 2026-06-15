# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RoboTUM's Forrest robot."""

from __future__ import annotations

import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.tendons.parameter_loader import ActuatorParameters, ForrestParameterConfig, load_forrest_parameter_config

# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

"""Configuration for RoboTUM's Forrest robot."""


def _implicit_actuator_cfg(config: ActuatorParameters) -> ImplicitActuatorCfg:
    return ImplicitActuatorCfg(
        joint_names_expr=list(config.joint_names_expr),
        effort_limit_sim=config.effort_limit_sim,
        velocity_limit_sim=config.velocity_limit_sim,
        stiffness=config.stiffness,
        damping=config.damping,
    )


def get_forrest_cfg(config: ForrestParameterConfig) -> ArticulationCfg:
    robot = config.robot
    physics = config.physics
    actuation = config.actuation
    flexor_angle = float(np.deg2rad(actuation.flexor_angle_deg))

    return ArticulationCfg(
        prim_path=robot.prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.getcwd(), robot.usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=physics.rigid_body.disable_gravity,
                retain_accelerations=physics.rigid_body.retain_accelerations,
                linear_damping=physics.rigid_body.linear_damping,
                angular_damping=physics.rigid_body.angular_damping,
                max_linear_velocity=physics.rigid_body.max_linear_velocity,
                max_angular_velocity=physics.rigid_body.max_angular_velocity,
                max_depenetration_velocity=physics.rigid_body.max_depenetration_velocity,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=physics.articulation.enabled_self_collisions,
                solver_position_iteration_count=physics.articulation.solver_position_iteration_count,
                solver_velocity_iteration_count=physics.articulation.solver_velocity_iteration_count,
            ),
        ),
        soft_joint_pos_limit_factor=robot.soft_joint_pos_limit_factor,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(robot.initial_base_position),
            joint_pos={
                # Left leg
                "lp1_pantograph": 0.0,
                "l0_acetabulofemoral_roll": 0.0,
                "l1_acetabulofemoral_lateral": 0.0,
                "l2_pseudo_acetabulofemoral_flexion": 0.0,
                "l3b_femorotibial_back": 0.0,
                "l4b_intertarsal_back": 0.0,
                "l3f_femorotibial_front": 0.0,
                "l4f_intertarsal_front": 0.0,
                "l4p_intertarsal_pulley": 0.0,
                "l5_metatarsophalangeal": float(np.deg2rad(-19.9)),
                "l6_interphalangeal": float(np.deg2rad(25.0)),
                "l8_knee_flexor": flexor_angle,
                # Right leg
                "rp1_pantograph": 0.0,
                "r0_acetabulofemoral_roll": 0.0,
                "r1_acetabulofemoral_lateral": 0.0,
                "r2_pseudo_acetabulofemoral_flexion": 0.0,
                "r3b_femorotibial_back": 0.0,
                "r4b_intertarsal_back": 0.0,
                "r3f_femorotibial_front": 0.0,
                "r4f_intertarsal_front": 0.0,
                "r4p_intertarsal_pulley": 0.0,
                "r5_metatarsophalangeal": float(np.deg2rad(-19.9)),
                "r6_interphalangeal": float(np.deg2rad(25.0)),
                "r8_knee_flexor": flexor_angle,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "pantograph": _implicit_actuator_cfg(actuation.pantograph),
            "hip_swing": _implicit_actuator_cfg(actuation.hip_swing),
            "hip_roll": _implicit_actuator_cfg(actuation.hip_roll),
            "hip_lateral": _implicit_actuator_cfg(actuation.hip_lateral),
            "knee_flex": _implicit_actuator_cfg(actuation.knee_flex),
        },
    )


FORREST_PARAMS = load_forrest_parameter_config()
FORREST_CFG = get_forrest_cfg(FORREST_PARAMS)
