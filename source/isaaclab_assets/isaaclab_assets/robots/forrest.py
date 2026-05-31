# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RoboTUM's Forrest robot."""

from __future__ import annotations
from pathlib import Path


import os
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

actuator_stiffness = 200
actuator_damping = 0.1

"""Configuration for RoboTUM's Forrest robot."""

FORREST_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/forrest_urdf_latest",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), "symlinks/forrest_urdf_latest/forrest_urdf_latest.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.45),
        joint_pos={
            # Left leg
            "rp2_pantograph": 0.0,
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
            "l8_knee_flexor": 0.0,
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
            "r8_knee_flexor": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Spring: Rest 63,5mm, Compressed 20mm, => travel 43,5mm 128 N/mm
        "pantograph": ImplicitActuatorCfg(
            joint_names_expr=[
                "rp1_pantograph",
                "rp2_pantograph",  # TODO: rename and create a new URDF
            ],
            effort_limit_sim=1e6,
            velocity_limit_sim=100.0,
            stiffness=128e3,
            damping=0.1,
        ),
        "hip_swing": ImplicitActuatorCfg(
            joint_names_expr=[
                "r2_pseudo_acetabulofemoral_flexion",
                "l2_pseudo_acetabulofemoral_flexion",
            ],
            effort_limit_sim=10.0e6,
            velocity_limit_sim=100.0,
            stiffness=actuator_stiffness,
            damping=actuator_damping,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                "r0_acetabulofemoral_roll",
                "l0_acetabulofemoral_roll",
            ],
            effort_limit_sim=10.0e6,
            velocity_limit_sim=100.0,
            stiffness=actuator_stiffness,
            damping=actuator_damping,
        ),
        "hip_lateral": ImplicitActuatorCfg(
            joint_names_expr=[
                "r1_acetabulofemoral_lateral",
                "l1_acetabulofemoral_lateral",
            ],
            effort_limit_sim=10.0e6,
            velocity_limit_sim=100.0,
            stiffness=actuator_stiffness * 10,
            damping=actuator_damping,
        ),
        "knee_flex": ImplicitActuatorCfg(
            joint_names_expr=[
                "r8_knee_flexor",
                "l8_knee_flexor",
            ],
            effort_limit_sim=10.0e6,
            velocity_limit_sim=100.0,
            stiffness=actuator_stiffness,
            damping=actuator_damping,
        ),
    },
)
