# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RoboTUM's Forrest robot."""

from __future__ import annotations
from pathlib import Path


import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

FORREST_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Forrest_URDF",
    spawn=sim_utils.UsdFileCfg(
    usd_path=os.path.join(os.getcwd(), "symlinks/Forrest_URDF/Forrest_URDF.usd"),
        # usd_path="symlinks/tron/robot.usd",
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
        pos=(0.0, 0.0, 1.5),
        # rot=(0.0, 0.0, 0.707, 0.707),   # <— quaternion for upright base
        joint_pos={
            # Left leg
            "l0_acetabulofemoral_roll": 0.0,
            "l1_acetabulofemoral_lateral": 0.0,
            "l2_pseudo_acetabulofemoral_flexion": 0.0,
            "l3b_femorotibial_back": 0.0,
            "l4b_intertarsal_back": 0.0,
            "l3f_femorotibial_front": 0.0,
            "l4f_intertarsal_front": 0.0,
            "l4p_intertarsal_pulley": 0.0,
            "l2p_acetabulofemoral_pulley": 0.0,
            "l2b_acetabulofemoral_flexion": 0.0,
            "l2f_acetabulofemoral_flexion": 0.0,
            "l5_metatarsophalangeal": 0.0,
            "l6_interphalangeal": 0.0,
            # Right leg
            "r0_acetabulofemoral_roll": 0.0,
            "r1_acetabulofemoral_lateral": 0.0,
            "r2_pseudo_acetabulofemoral_flexion": 0.0,
            "r3b_femorotibial_back": 0.0,
            "r4b_intertarsal_back": 0.0,
            "r3f_femorotibial_front": 0.0,
            "r4f_intertarsal_front": 0.0,
            "r4p_intertarsal_pulley": 0.0,
            "r2p_acetabulofemoral_pulley": 0.0,
            "r2b_acetabulofemoral_flexion": 0.0,
            "r2f_acetabulofemoral_flexion": 0.0,
            "r5_metatarsophalangeal": 0.0,
            "r6_interphalangeal": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
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
            # stiffness=42.0,
            stiffness=100.0,
            damping=2.5,
        )
    },
)
"""Configuration for RoboTUM's Forrest robot."""
