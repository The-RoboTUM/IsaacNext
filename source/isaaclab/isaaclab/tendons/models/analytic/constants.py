# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constants for our tendon model."""

import numpy as np
import torch
from dataclasses import dataclass, field

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.utils import list_from_dict

N_CHAIN_LINKS_PER_LEG: int = 6  # -> number of links in the kinematic chain of the leg
N_CONNECTOR_OFFSETS: int = 6  # -> GST, DFT, EDT2, KFT have a connector not on the ji-ji+1 axis; EDT1 has two
N_LINK_LENGTHS_PER_LEG: int = (
    N_CHAIN_LINKS_PER_LEG
    + N_CONNECTOR_OFFSETS
    # -> number of link lengths per leg (including virtual links for tendon attachment points)
)
N_JOINTS: int = 5
N_RADII: int = 11  # -> number of pulley radii

N_TENDON_SECTION_LENGTHS: int = (
    11  # lengths between two pulleys in contact, or between pulley and tendon attachment point if only one pulley
)
N_TENDON_THETA_OFFSETS: int = (
    N_JOINTS + N_CONNECTOR_OFFSETS
)  # -> number of raw joint angle offsets: 5 for joints, 6 for connectors
N_Q_OFFSETS: int = 6  # -> number of joint angle theta offsets to wrapping angles : 4 for GST, 2 for DFT
N_QHAT_OFFSETS: int = (
    1  # -> number of joint angle theta hat offsets to wrapping angles : 4 for GST, 2 for DFT, j6 for EDT2
)

N_TENDON_TANGENCY_ANGLES: int = 9  # number of fixed tendon tangency angles used in the computation

JOINT_AXIS_IDX: int = 0  # axis index for joint torques around x-axis

dev = "cuda"

link_names_right = list_from_dict(
    {
        tids.I_CHAIN_LINK_23: "outside_hip_v2_assy_axial_1",  # "knee_assyv9_1",  # 23
        tids.I_CHAIN_LINK_34: "s12_front_assy_1",  # "s12_front_assyv6_1",  # 34
        tids.I_CHAIN_LINK_4prime5: "s23_assy_1",  # "s23_assyv18_1",  # 4'5
        tids.I_CHAIN_LINK_56: "s34_foot_connector_assy_1",  # "s34_foot_connector_assyv20_1",  # 56
        tids.I_CHAIN_LINK_67: "s45_digit_assy_1",  # "s45_digit_assyv2_1",  # 67
        tids.I_CHAIN_LINK_38: "knee_motor_winch_big_motor_1",  # j8 to end of pulley (p9), better name would be 89
    },
    N_CHAIN_LINKS_PER_LEG,
)
link_names_left = list_from_dict(
    {
        tids.I_CHAIN_LINK_23: "outside_hip_v2_assy_axial_left_1",  # 23 # is also 28 if you think about it
        tids.I_CHAIN_LINK_34: "s12_front_assy_2",  # 34
        tids.I_CHAIN_LINK_4prime5: "s23_assy_2",  # 4'5
        tids.I_CHAIN_LINK_56: "s34_foot_connector_assy_2",  # 56
        tids.I_CHAIN_LINK_67: "s45_digit_assy_2",  # 67
        tids.I_CHAIN_LINK_38: "knee_motor_winch_big_motor_2",  # j8 to end of pulley (p9), better name would be 89
    },
    N_CHAIN_LINKS_PER_LEG,
)
joint_names_right = list_from_dict(
    {
        tids.I_JOINT_3: "r3f_femorotibial_front",  # j3
        tids.I_JOINT_4: "r4p_intertarsal_pulley",  # j4
        tids.I_JOINT_5: "r5_metatarsophalangeal",  # j5
        tids.I_JOINT_6: "r6_interphalangeal",  # j6
        tids.I_JOINT_8: "r8_knee_flexor",  # j8
    },
    N_JOINTS,
)
joint_names_left = list_from_dict(
    {
        tids.I_JOINT_3: "l3f_femorotibial_front",  # j3
        tids.I_JOINT_4: "l4p_intertarsal_pulley",  # j4
        tids.I_JOINT_5: "l5_metatarsophalangeal",  # j5
        tids.I_JOINT_6: "l6_interphalangeal",  # j6
        tids.I_JOINT_8: "l8_knee_flexor",  # j8
    },
    N_JOINTS,
)

hip_joint_names = [
    "l0_acetabulofemoral_roll",
    "l1_acetabulofemoral_lateral",
    "l2_pseudo_acetabulofemoral_flexion",
    "r0_acetabulofemoral_roll",
    "r1_acetabulofemoral_lateral",
    "r2_pseudo_acetabulofemoral_flexion",
]

actuated_joint_names = [
    "l0_acetabulofemoral_roll",
    "l1_acetabulofemoral_lateral",
    "l2_pseudo_acetabulofemoral_flexion",
    "r0_acetabulofemoral_roll",
    "r1_acetabulofemoral_lateral",
    "r2_pseudo_acetabulofemoral_flexion",
    "l8_knee_flexor",
    "r8_knee_flexor",
]

all_joint_names_right = [
    "r0_acetabulofemoral_roll,"  # j0, position/torque control
    "r1_acetabulofemoral_lateral",  # j1, position/torque control
    "rp1_pantograph",
    # pantograph, actuated but always set to 0.0, stiffness? blockhöhe?     "s12p_pantograph_spring_assy_topv2_1" -> "s12p_pantograph_spring_assy_botv1_1"
    "r2_pseudo_acetabulofemoral_flexion",
    # j2 -> position control, stiffness? damping?       "outside_hip_v2_assyv28_1" -> "knee_assyv9_1"
    "r3b_femorotibial_back",
    # excluded from articulation (fourbar), between j2 and j3        "knee_assyv9_1" -> "s12p_pantograph_spring_assy_topv2_1"
    "r3f_femorotibial_front",
    # j3 -> torque control, applied alongside other tendon torques  "knee_assyv9_1" -> "s12_front_assyv6_1"
    "r4f_intertarsal_front",
    # only shows the pulley position q4' -> fix                      "s12_front_assyv6_1" -> "main_gst_pully_assyv4_1"
    "r4b_intertarsal_back",
    # not actuated (fourbar), above j4                                "s12p_pantograph_spring_assy_botv1_1" -> "s23_assyv18_1_virtual"
    "r4p_intertarsal_pulley",
    # j4, not actuated but affected by tendon                       "s12_front_assyv6_1" -> "s23_assyv18_1"
    "r5_metatarsophalangeal",
    # j5, not actuated but affected by tendon                       "s23_assyv18_1" -> "s34_foot_connector_assyv20_1"
    "r6_interphalangeal",
    # j6, not actuated but affected by tendon                           "s34_foot_connector_assyv20_1" -> "s45_digit_assyv2_1"
    "virtual_s23_assyv18_1_anchor",
    # necessary for the urdf exporter but not actuated        "s23_assyv18_1" -> "s23_assyv18_1_virtual"
    "r8_knee_flexor",  # j8, position/torque control
]


# TODO: verify tendon lengths when prototype is built
@dataclass
class TendonConstants:
    """Fixed baseline mathematical constants for our tendon model: link lengths and pulley radii etc."""

    # gst_stiffness: float = 128e3 * 0  # FIXME: Reduced for simulation stability by 10x
    n = 3
    gst_stiffness: float = 2 * (10 ** (n + 2)) * 1
    dft_stiffness: float = 5 * (10 ** (n + 1)) * 1  # FIXME: find out real value
    edt1_stiffness: float = 5 * (10 ** (n + 2)) * 1  # FIXME: find out real value
    edt2_stiffness: float = 5 * (10 ** (n + 2)) * 1  # FIXME: find out real value
    kft_stiffness: float = 5 * (10 ** (n + 2)) * 1  # FIXME: find out real value

    gst_spring_rest_length: float = 0.06
    upper_gst_length: float = 0.6367 + 0.055  # FIXME: measure correct value
    lower_gst_length: float = 0.6314  # FIXME: measure correct value
    # dft_length: float = 0.345  # FIXME: measure correct value
    dft_length: float = 0.345 + 0.039  # FIXME: measure correct value
    edt1_length: float = 0.55 - 0.01  # FIXME: measure correct value
    edt2_length: float = 0.66 - 0.01  # FIXME: measure correct value
    # edt1_length: float = 0.48  # FIXME: measure correct value
    # edt2_length: float = 0.58  # FIXME: measure correct value
    kft_length: float = 0.402 + 0.05  # FIXME: measure correct value

    joint_offsets_theta: torch.Tensor = torch.deg2rad(  # between joint-to-joint links, offsets to raw joint angles
        torch.tensor(
            list_from_dict(
                {
                    tids.I_JOINT_3: 227.671,
                    tids.I_JOINT_4: 225.931,
                    tids.I_JOINT_5: 180.0,
                    tids.I_JOINT_6: 270.0,
                    tids.I_JOINT_8: 180.0,
                },
                N_JOINTS,
            ),
            device=dev,
        )
    )
    joint_directions: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_JOINT_3: -1.0,
                tids.I_JOINT_4: +1.0,
                tids.I_JOINT_5: -1.0,
                tids.I_JOINT_6: -1.0,
                tids.I_JOINT_8: -1.0,
            },
            N_JOINTS,
        ),
        device=dev,
    )
    pulley_radii: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_RADIUS_GST_3: 0.029,
                tids.I_RADIUS_GST_4: 0.1,
                tids.I_RADIUS_GST_4prime: 0.05,
                tids.I_RADIUS_GST_5: 0.04,
                tids.I_RADIUS_GST_6: 0.01,
                tids.I_RADIUS_DFT_5: 0.04,
                tids.I_RADIUS_DFT_6: 0.01,
                tids.I_RADIUS_EDT1_5: 0.04,
                tids.I_RADIUS_EDT2_5: 0.04,
                tids.I_RADIUS_EDT2_6: 0.01,
                tids.I_RADIUS_KFT_8: 0.035,
            },
            N_RADII,
        ),
        device=dev,
    )
    chain_link_lengths: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_CHAIN_LINK_23: 0.33,
                tids.I_CHAIN_LINK_38: 0.33,
                tids.I_CHAIN_LINK_34: 0.461,
                tids.I_CHAIN_LINK_4prime5: 0.357,
                tids.I_CHAIN_LINK_56: 0.165,
                tids.I_CHAIN_LINK_67: 0.044,
            },
            N_CHAIN_LINKS_PER_LEG,
        ),
        device=dev,
    )
    connector_link_lengths_longitudinal: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_CONNECTOR_LINK_GST_23: 0.1072,
                tids.I_CONNECTOR_LINK_DFT_C5: 0.13,
                tids.I_CONNECTOR_LINK_EDT1_C4: 0.17,
                tids.I_CONNECTOR_LINK_EDT1_5C: 0.088,
                tids.I_CONNECTOR_LINK_EDT2_C4: 0.22,
                tids.I_CONNECTOR_LINK_KFT_3C: 0.0635,
            },
            N_CONNECTOR_OFFSETS,
        )
    )
    connector_link_lengths_lateral: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_CONNECTOR_LINK_GST_23: 0.0,
                tids.I_CONNECTOR_LINK_DFT_C5: 0.04,
                tids.I_CONNECTOR_LINK_EDT1_C4: 0.04,
                tids.I_CONNECTOR_LINK_EDT1_5C: 0.007,  # measured towards the front side (GST spring side) TODO: verify
                tids.I_CONNECTOR_LINK_EDT2_C4: 0.04,
                tids.I_CONNECTOR_LINK_KFT_3C: 0.009,  # measured from the j3-j4 axis towards the GST spring
            },
            N_CONNECTOR_OFFSETS,
        )
    )
    gst_phi_23_j3: float = float(
        np.deg2rad(98.874)
    )  # angle between link 23 and line from joint 3 to the tangency point of the GST on pulley 3
    angle_4prime5_to_j44prime = np.deg2rad(124.069)  # angle between link 4'5 and line from joint 4 to 4-4' transition


@dataclass
class TendonConstantRandomizationRanges:
    """Ranges for randomizing tendon constants for sim-to-real transfer."""

    # gst_stiffness: tuple[float, float] = (-10e3, 10e3)
    # dft_stiffness: tuple[float, float] = (-10e3, 10e3)
    # edt1_stiffness: tuple[float, float] = (-100e3, 100e3)
    # edt2_stiffness: tuple[float, float] = (-100e3, 100e3)
    # kft_stiffness: tuple[float, float] = (-100e3, 100e3)

    gst_stiffness: tuple[float, float] = (0.0, 0.0)
    dft_stiffness: tuple[float, float] = (0.0, 0.0)
    edt1_stiffness: tuple[float, float] = (0.0, 0.0)
    edt2_stiffness: tuple[float, float] = (0.0, 0.0)
    kft_stiffness: tuple[float, float] = (0.0, 0.0)

    # dft_length: tuple[float, float] = (-0.005, 0.005)
    # edt1_length: tuple[float, float] = (-0.005, 0.005)
    # edt2_length: tuple[float, float] = (-0.005, 0.005)
    # kft_length: tuple[float, float] = (-0.005, 0.005)
    dft_length: tuple[float, float] = (-0.000, 0.000)
    edt1_length: tuple[float, float] = (-0.000, 0.00)
    edt2_length: tuple[float, float] = (-0.00, 0.00)
    kft_length: tuple[float, float] = (-0.00, 0.00)

    # gst_spring_rest_length: tuple[float, float] = (-0.005, 0.005)
    # upper_gst_length: tuple[float, float] = (-0.01, 0.01)
    # lower_gst_length: tuple[float, float] = (-0.01, 0.01)

    gst_spring_rest_length: tuple[float, float] = (-0.00, 0.00)
    upper_gst_length: tuple[float, float] = (-0.0, 0.0)
    lower_gst_length: tuple[float, float] = (-0.0, 0.0)

    joint_offsets_theta: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_JOINT_3: (-0.05, 0.05),
                tids.I_JOINT_4: (-0.05, 0.05),
                tids.I_JOINT_5: (-0.05, 0.05),
                tids.I_JOINT_6: (-0.05, 0.05),
                tids.I_JOINT_8: (-0.05, 0.05),
            },
            N_JOINTS,
        )
    )

    pulley_radii: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_RADIUS_GST_3: (-0.001, 0.001),
                tids.I_RADIUS_GST_4: (-0.001, 0.001),
                tids.I_RADIUS_GST_4prime: (-0.001, 0.001),
                tids.I_RADIUS_GST_5: (-0.001, 0.001),
                tids.I_RADIUS_GST_6: (-0.001, 0.001),
                tids.I_RADIUS_DFT_5: (-0.001, 0.001),
                tids.I_RADIUS_DFT_6: (-0.001, 0.001),
                tids.I_RADIUS_EDT1_5: (-0.001, 0.001),
                tids.I_RADIUS_EDT2_5: (-0.001, 0.001),
                tids.I_RADIUS_EDT2_6: (-0.001, 0.001),
                tids.I_RADIUS_KFT_8: (-0.001, 0.001),
            },
            N_RADII,
        )
    )

    chain_link_lengths: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_CHAIN_LINK_23: (-0.001, 0.001),
                tids.I_CHAIN_LINK_38: (-0.001, 0.001),
                tids.I_CHAIN_LINK_34: (-0.001, 0.001),
                tids.I_CHAIN_LINK_4prime5: (-0.001, 0.001),
                tids.I_CHAIN_LINK_56: (-0.001, 0.001),
                tids.I_CHAIN_LINK_67: (-0.001, 0.001),
            },
            N_CHAIN_LINKS_PER_LEG,
        )
    )
    connector_link_lengths_longitudinal: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_CONNECTOR_LINK_GST_23: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_DFT_C5: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_EDT1_C4: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_EDT1_5C: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_EDT2_C4: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_KFT_3C: (-0.001, 0.001),
            },
            N_CONNECTOR_OFFSETS,
        )
    )
    connector_link_lengths_lateral: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_CONNECTOR_LINK_GST_23: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_DFT_C5: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_EDT1_C4: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_EDT1_5C: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_EDT2_C4: (-0.001, 0.001),
                tids.I_CONNECTOR_LINK_KFT_3C: (-0.001, 0.001),
            },
            N_CONNECTOR_OFFSETS,
        )
    )
    gst_phi_23_j3: tuple[float, float] = (-0.05, 0.05)
    angle_4prime5_to_j44prime: tuple[float, float] = (-0.05, 0.05)


dummy_randomization = TendonConstantRandomizationRanges(
    gst_stiffness=(0.0, 0.0),
    dft_stiffness=(0.0, 0.0),
    edt1_stiffness=(0.0, 0.0),
    edt2_stiffness=(0.0, 0.0),
    kft_stiffness=(0.0, 0.0),
    dft_length=(0.0, 0.0),
    edt1_length=(0.0, 0.0),
    edt2_length=(0.0, 0.0),
    kft_length=(0.0, 0.0),
    gst_spring_rest_length=(0.0, 0.0),
    upper_gst_length=(0.0, 0.0),
    lower_gst_length=(0.0, 0.0),
    joint_offsets_theta=list_from_dict(
        {
            tids.I_JOINT_3: (0.0, 0.0),
            tids.I_JOINT_4: (0.0, 0.0),
            tids.I_JOINT_5: (0.0, 0.0),
            tids.I_JOINT_6: (0.0, 0.0),
            tids.I_JOINT_8: (0.0, 0.0),
        },
        N_JOINTS,
    ),
    pulley_radii=list_from_dict(
        {
            tids.I_RADIUS_GST_3: (0.0, 0.0),
            tids.I_RADIUS_GST_4: (0.0, 0.0),
            tids.I_RADIUS_GST_4prime: (0.0, 0.0),
            tids.I_RADIUS_GST_5: (0.0, 0.0),
            tids.I_RADIUS_GST_6: (0.0, 0.0),
            tids.I_RADIUS_DFT_5: (0.0, 0.0),
            tids.I_RADIUS_DFT_6: (0.0, 0.0),
            tids.I_RADIUS_EDT1_5: (0.0, 0.0),
            tids.I_RADIUS_EDT2_5: (0.0, 0.0),
            tids.I_RADIUS_EDT2_6: (0.0, 0.0),
            tids.I_RADIUS_KFT_8: (0.0, 0.0),
        },
        N_RADII,
    ),
    chain_link_lengths=list_from_dict(
        {
            tids.I_CHAIN_LINK_23: (0.0, 0.0),
            tids.I_CHAIN_LINK_38: (0.0, 0.0),
            tids.I_CHAIN_LINK_34: (0.0, 0.0),
            tids.I_CHAIN_LINK_4prime5: (0.0, 0.0),
            tids.I_CHAIN_LINK_56: (0.0, 0.0),
            tids.I_CHAIN_LINK_67: (0.0, 0.0),
        },
        N_CHAIN_LINKS_PER_LEG,
    ),
    connector_link_lengths_longitudinal=list_from_dict(
        {
            tids.I_CONNECTOR_LINK_GST_23: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_DFT_C5: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_EDT1_C4: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_EDT1_5C: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_EDT2_C4: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_KFT_3C: (0.0, 0.0),
        },
        N_CONNECTOR_OFFSETS,
    ),
    connector_link_lengths_lateral=list_from_dict(
        {
            tids.I_CONNECTOR_LINK_GST_23: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_DFT_C5: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_EDT1_C4: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_EDT1_5C: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_EDT2_C4: (0.0, 0.0),
            tids.I_CONNECTOR_LINK_KFT_3C: (0.0, 0.0),
        },
        N_CONNECTOR_OFFSETS,
    ),
    gst_phi_23_j3=(0.0, 0.0),
    angle_4prime5_to_j44prime=(0.0, 0.0),
)
