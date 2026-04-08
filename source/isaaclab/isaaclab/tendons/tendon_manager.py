"""Tendon manager implementation for all tendons."""

from isaaclab.utils.math import quat_apply_inverse, quat_rotate_inverse
import numpy as np
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.tendons.constants import (
    TendonConstantRandomizationRanges,
    dummy_randomization,
    link_names_left,
    link_names_right,
    joint_names_left,
    joint_names_right,
    N_CHAIN_LINKS_PER_LEG,
    JOINT_AXIS_IDX,
)
from isaaclab.tendons.tendon_data import TendonData, TendonDataJIT

import isaaclab.tendons.indices as tids


# todo comment
@torch.jit.script
def angle_from_sws(
    a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    x = a - b * torch.cos(theta)
    y = b * torch.sin(theta)
    return torch.atan2(y, x)


@torch.jit.script
def compute_delta_l_s_jit(
    joint_angles: torch.Tensor, tendon_data: TendonDataJIT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 0) transform joint angles to thetas and qs
    joint_angles_signed = tendon_data.joint_directions * joint_angles
    thetas = torch.empty_like(tendon_data.tendon_offsets_theta)
    thetas[
        (
            tids.I_THETA_GST_3,
            tids.I_THETA_GST_4,
            tids.I_THETA_GST_5,
            tids.I_THETA_ALL_6,
            tids.I_THETA_DFT_5,
            tids.I_THETA_EDT1_4,
            tids.I_THETA_EDT1_5,
            tids.I_THETA_EDT2_4,
            tids.I_THETA_EDT2_5,
            tids.I_THETA_KFT_3,
            tids.I_THETA_KFT_8,
        )
    ] = (
        joint_angles_signed[
            (
                tids.I_JOINT_3,
                tids.I_JOINT_4,
                tids.I_JOINT_5,
                tids.I_JOINT_6,
                tids.I_JOINT_5,
                tids.I_JOINT_4,
                tids.I_JOINT_5,
                tids.I_JOINT_4,
                tids.I_JOINT_5,
                tids.I_JOINT_3,
                tids.I_JOINT_8,
            )
        ]
        + tendon_data.tendon_offsets_theta[
            (
                tids.I_THETA_GST_3,
                tids.I_THETA_GST_4,
                tids.I_THETA_GST_5,
                tids.I_THETA_ALL_6,
                tids.I_THETA_DFT_5,
                tids.I_THETA_EDT1_4,
                tids.I_THETA_EDT1_5,
                tids.I_THETA_EDT2_4,
                tids.I_THETA_EDT2_5,
                tids.I_THETA_KFT_3,
                tids.I_THETA_KFT_8,
            )
        ]
    )
    qs = torch.empty_like(tendon_data.tendon_offsets_q_theta)
    qs[
        (
            tids.I_Q_GST_3,
            tids.I_Q_GST_4,
            tids.I_Q_GST_5,
            tids.I_Q_GST_6,
            tids.I_Q_DFT_5,
            tids.I_Q_DFT_6,
        )
    ] = (
        thetas[
            (
                tids.I_THETA_GST_3,
                tids.I_THETA_GST_4,
                tids.I_THETA_GST_5,
                tids.I_THETA_ALL_6,
                tids.I_THETA_DFT_5,
                tids.I_THETA_ALL_6,
            )
        ]
        + tendon_data.tendon_offsets_q_theta[
            (
                tids.I_Q_GST_3,
                tids.I_Q_GST_4,
                tids.I_Q_GST_5,
                tids.I_Q_GST_6,
                tids.I_Q_DFT_5,
                tids.I_Q_DFT_6,
            )
        ]
    )

    theta_hats = -thetas + 2 * torch.pi
    qhats = torch.empty_like(tendon_data.tendon_offsets_qhat_thetahat)
    qhats[(tids.I_QHAT_EDT2_6,)] = (
        theta_hats[(tids.I_THETA_ALL_6,)]
        + tendon_data.tendon_offsets_qhat_thetahat[[(tids.I_QHAT_EDT2_6,)]]
    )

    ### --------------- GST --------------- ###
    # 1) evaluate conditions
    # 1a) compute h5^B
    GST_x_4prime6_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * torch.cos(thetas[:, tids.I_THETA_GST_5])
    )
    GST_x_4prime6 = torch.sqrt(GST_x_4prime6_squared)
    GST_l_4prime6_squared = GST_x_4prime6_squared - (
        (
            tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime]
            - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
        )
        ** 2
    )
    GST_l_4prime6 = torch.sqrt(GST_l_4prime6_squared)
    GST_phi_4prime_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.I_THETA_GST_5],
    )

    GST_phi_4prime_b = torch.acos(
        (
            tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_4prime]
            + GST_x_4prime6_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_6]
            - GST_l_4prime6_squared
        )
        / (2 * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] * GST_x_4prime6)
    )
    GST_phi_4prime_B = GST_phi_4prime_a + GST_phi_4prime_b
    GST_h5_B = tendon_data.pulley_radii[
        :, tids.I_RADIUS_GST_4prime
    ] - tendon_data.link_lengths[:, tids.I_LINK_4prime5] * torch.cos(GST_phi_4prime_B)

    # 1b) compute h5^C and h6^C
    GST_theta_6_a = torch.pi - thetas[:, tids.I_THETA_GST_5] - GST_phi_4prime_a
    GST_theta_6_b = thetas[:, tids.I_THETA_ALL_6] - GST_theta_6_a
    GST_x_4prime7_squared = (
        GST_x_4prime6_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2
        * GST_x_4prime6
        * tendon_data.link_lengths[:, tids.I_LINK_67]
        * torch.cos(GST_theta_6_b)
    )
    GST_x_4prime7 = torch.sqrt(GST_x_4prime7_squared)
    GST_phi_4prime_d = angle_from_sws(
        GST_x_4prime6,
        tendon_data.link_lengths[:, tids.I_LINK_67],
        GST_theta_6_b,
    )

    GST_phi_4prime_c = torch.acos(
        tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] / GST_x_4prime7
    )
    GST_phi_4prime_C = GST_phi_4prime_a + GST_phi_4prime_c + GST_phi_4prime_d
    GST_h5_C = tendon_data.pulley_radii[
        :, tids.I_RADIUS_GST_4prime
    ] - tendon_data.link_lengths[:, tids.I_LINK_4prime5] * torch.cos(GST_phi_4prime_C)
    GST_h6_C = tendon_data.pulley_radii[
        :, tids.I_RADIUS_GST_4prime
    ] - GST_x_4prime6 * torch.cos(GST_phi_4prime_c + GST_phi_4prime_d)

    # print("Theta 6:", thetas[:, self.JOINT_ANGLES_6])

    # 1c) compute h6^D
    GST_x_57_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * tendon_data.link_lengths[:, tids.I_LINK_67]
        * torch.cos(
            thetas[:, tids.I_THETA_ALL_6],
        )
    )
    GST_x_57 = torch.sqrt(GST_x_57_squared)
    GST_l_57_squared = (
        GST_x_57_squared - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] ** 2
    )
    GST_l_57 = torch.sqrt(GST_l_57_squared)

    GST_phi_5_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_67],
        thetas[:, tids.I_THETA_ALL_6],
    )

    GST_phi_5_b = torch.acos(
        tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] / GST_x_57
    )
    GST_phi_5_D = GST_phi_5_a + GST_phi_5_b
    GST_h6_D = tendon_data.pulley_radii[
        :, tids.I_RADIUS_GST_5
    ] - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.cos(GST_phi_5_D)

    GST_h5_B_disengaged = torch.where(
        GST_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5],
        True,
        False,
    )
    GST_h5_C_disengaged = torch.where(
        GST_h5_C > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5],
        True,
        False,
    )
    GST_h6_C_disengaged = torch.where(
        GST_h6_C > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6],
        True,
        False,
    )

    GST_h6_D_disengaged = torch.where(
        GST_h6_D > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6],
        True,
        False,
    )
    GST_state_C = (GST_h5_B_disengaged & GST_h6_C_disengaged) | (
        GST_h6_D_disengaged & GST_h5_C_disengaged
    )
    GST_state_B = ~GST_state_C & GST_h5_B_disengaged
    GST_state_D = ~GST_state_C & GST_h6_D_disengaged
    assert (
        GST_state_B.sum() + GST_state_D.sum() == (GST_state_B | GST_state_D).sum()
    ), "States B and D are active simultaneously"
    GST_state_A = ~(GST_state_B | GST_state_C | GST_state_D)

    # 2) compute energy with conditional function for lower tendon state length
    # state A
    GST_lower_tendon_state_length_after_4prime_A = (
        tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5]
        + qs[:, tids.I_Q_GST_5] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_56]
        + qs[:, tids.I_Q_GST_6] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_67]
    )

    # state B
    GST_q6_B = (
        thetas[:, tids.I_THETA_ALL_6]
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_GST_67_J6]
        - 2 * torch.pi
        + GST_phi_4prime_B
        + thetas[:, tids.I_THETA_GST_5]
    )

    GST_lower_tendon_state_length_after_4prime_B = (
        GST_l_4prime6
        + GST_q6_B * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_67]
    )

    # state C
    GST_l_4prime7_squared = (
        GST_x_4prime7_squared
        - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_4prime]
    )
    GST_l_4prime7 = torch.sqrt(GST_l_4prime7_squared)
    GST_lower_tendon_state_length_after_4prime_C = GST_l_4prime7

    # state D
    GST_q5_D = (
        thetas[:, tids.I_THETA_GST_5]
        - tendon_data.tendon_tangency_angles[
            :, tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J5
        ]
        - GST_phi_5_D
    )
    GST_lower_tendon_state_length_after_4prime_D = (
        tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5]
        + GST_q5_D * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
        + GST_l_57
    )

    GST_lower_tendon_state_length_after_4prime = torch.where(
        GST_state_A,
        GST_lower_tendon_state_length_after_4prime_A,
        torch.where(
            GST_state_B,
            GST_lower_tendon_state_length_after_4prime_B,
            torch.where(
                GST_state_C,
                GST_lower_tendon_state_length_after_4prime_C,
                GST_lower_tendon_state_length_after_4prime_D,
            ),
        ),
    )

    GST_q4prime = (
        tendon_data.lower_gst_length - GST_lower_tendon_state_length_after_4prime
    ) / tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime]

    GST_q4_base = (
        thetas[:, tids.I_THETA_GST_4]
        - GST_q4prime
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_GST_34_J4]
    )

    # Use torch.where instead of in-place masked operations to preserve gradient flow
    GST_q4_adjustment = torch.where(
        torch.logical_or(GST_state_A, GST_state_D),
        tendon_data.tendon_tangency_angles[
            :, tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J4
        ],
        torch.where(
            GST_state_B,
            GST_phi_4prime_B,
            GST_phi_4prime_C,  # state_C is the only remaining case
        ),
    )
    GST_q4 = GST_q4_base - GST_q4_adjustment

    GST_delta_L_s = (
        tendon_data.upper_gst_length
        - tendon_data.gst_spring_rest_length
        - tendon_data.link_lengths[:, tids.I_LINK_GST_23]
        - qs[:, tids.I_Q_GST_3] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_3]
        - tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_34]
        - GST_q4 * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4]
    )

    ### --------- DFT ---------- ###
    DFT_q5 = qs[:, tids.I_Q_DFT_5]
    DFT_q6 = qs[:, tids.I_Q_DFT_6]
    DFT_l_c5 = tendon_data.tendon_section_lengths[
        :, tids.I_TENDON_SECTION_LENGTH_DFT_C5
    ]
    DFT_l_56 = tendon_data.tendon_section_lengths[
        :, tids.I_TENDON_SECTION_LENGTH_DFT_56
    ]
    DFT_l_6c = tendon_data.tendon_section_lengths[
        :, tids.I_TENDON_SECTION_LENGTH_DFT_6C
    ]

    DFT_delta_L_s = (
        tendon_data.dft_length
        - DFT_l_c5
        - DFT_q5 * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5]
        - DFT_l_56
        - DFT_q6 * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6]
        - DFT_l_6c
    )

    ### --------- KFT ---------- ###
    KFT_l_8c_j_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_83]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_KFT_3C]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_83]
        * tendon_data.link_lengths[:, tids.I_LINK_KFT_3C]
        * torch.cos(theta_hats[:, tids.I_THETA_KFT_3])
    )
    KFT_l_8c = torch.sqrt(
        KFT_l_8c_j_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_KFT_8]
    )
    KFT_phi_8 = torch.atan2(KFT_l_8c, tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8])
    KFT_phi_8_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_83],
        tendon_data.link_lengths[:, tids.I_LINK_KFT_3C],
        theta_hats[:, tids.I_THETA_KFT_3],
    )
    KFT_q8 = thetas[:, tids.I_THETA_KFT_8] - KFT_phi_8 + KFT_phi_8_a

    KFT_delta_L_s = (
        tendon_data.kft_length
        - KFT_q8 * tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8]
        - KFT_l_8c
    )

    ### ------------- EDT1 ------------- ###

    EDT1_x_c5_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_C4]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4]
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * torch.cos(theta_hats[:, tids.I_THETA_EDT1_4])
    )
    EDT1_x_c5 = torch.sqrt(EDT1_x_c5_squared)
    EDT1_phi_4_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT1_4],
    )
    EDT1_thetahat_5_a = torch.pi - theta_hats[:, tids.I_THETA_EDT1_4] - EDT1_phi_4_a

    # state A: tendon wraps around j5 pulley
    EDT1_l_c5_A = torch.sqrt(
        EDT1_x_c5_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT1_5]
    )
    EDT1_phi_45_A = torch.atan2(
        EDT1_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
    )
    EDT1_q5_A = (
        2 * torch.pi
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_EDT1_5C_J5]
        - thetas[:, tids.I_THETA_EDT1_5]
        - EDT1_thetahat_5_a
        - EDT1_phi_45_A
    )

    # state B: tendon does not wrap around j5 pulley

    EDT1_thetahat_5_b = theta_hats[:, tids.I_THETA_EDT1_5] - EDT1_thetahat_5_a
    EDT1_phi_4_b = angle_from_sws(
        EDT1_x_c5,
        tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C],
        EDT1_thetahat_5_b,
    )

    EDT1_h5_B = EDT1_x_c5 * torch.sin(EDT1_phi_4_b)
    EDT1_l_cc = torch.sqrt(
        EDT1_x_c5_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_5C]
        - 2
        * EDT1_x_c5
        * tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C]
        * torch.cos(EDT1_thetahat_5_b)
    )

    EDT1_state_B = EDT1_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
    EDT1_state_A = ~EDT1_state_B

    EDT1_L_s = torch.where(
        EDT1_state_B,
        EDT1_l_cc,
        EDT1_l_c5_A
        + EDT1_q5_A * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT1_5C],
    )

    EDT1_delta_L_s = tendon_data.edt1_length - EDT1_L_s

    ### ------------- EDT2 ------------- ###
    EDT2_x_c5_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * torch.cos(theta_hats[:, tids.I_THETA_EDT2_4])
    )
    EDT2_x_c5 = torch.sqrt(EDT2_x_c5_squared)
    EDT2_phi_4_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT2_4],
    )
    EDT2_thetahat_5_a = torch.pi - theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_phi_4_a

    # state A: tendon wraps around j5 and j6 pulleys
    EDT2_l_c5_A = torch.sqrt(
        EDT2_x_c5_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5]
    )
    EDT2_phi_45_A = torch.atan2(
        EDT2_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
    )
    EDT2_q5_A = (
        2 * torch.pi
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_EDT2_56_J5]
        - thetas[:, tids.I_THETA_EDT2_5]
        - EDT2_thetahat_5_a
        - EDT2_phi_45_A
    )
    EDT2_L_s_A = (
        EDT2_l_c5_A
        + EDT2_q5_A * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT2_56]
        + qhats[:, tids.I_QHAT_EDT2_6]
        * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT2_6C]
    )

    # state B: tendon wraps around j6 pulley but not j5 pulley
    EDT2_x_64prime_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * torch.cos(thetas[:, tids.I_THETA_EDT2_5])
    )
    EDT2_x_64prime = torch.sqrt(EDT2_x_64prime_squared)
    EDT2_phi_6_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT2_5],
    )
    EDT2_thetahat_4_a = torch.pi - theta_hats[:, tids.I_THETA_EDT2_5] - EDT2_phi_6_a
    EDT2_thetahat_4_b = theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_thetahat_4_a
    EDT2_x_6c_squared = (
        EDT2_x_64prime_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
        - 2
        * EDT2_x_64prime
        * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
        * torch.cos(EDT2_thetahat_4_b)
    )
    EDT2_x_6c = torch.sqrt(EDT2_x_6c_squared)
    EDT2_phi_6_d = angle_from_sws(
        EDT2_x_6c,
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        EDT2_thetahat_4_b,
    )
    EDT2_l_c6_B = torch.sqrt(
        EDT2_x_6c_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_6]
    )
    EDT2_phi_6_c = torch.atan2(
        EDT2_l_c6_B, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
    )
    EDT2_phi_6_B = EDT2_phi_6_a + EDT2_phi_6_c + EDT2_phi_6_d
    EDT2_q6_B = theta_hats[:, tids.I_THETA_ALL_6] - EDT2_phi_6_B
    EDT2_h5_B = tendon_data.pulley_radii[
        :, tids.I_RADIUS_EDT2_6
    ] - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.cos(EDT2_phi_6_B)
    EDT2_L_s_B = (
        EDT2_l_c6_B
        + EDT2_q6_B * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT2_6C]
    )

    # state C: tendon does not wrap around any pulley
    EDT2_l_46_j_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * torch.cos(thetas[:, tids.I_THETA_EDT2_5])
    )
    EDT2_l_46_j = torch.sqrt(EDT2_l_46_j_squared)
    EDT2_gamma_4 = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.I_THETA_EDT2_5],
    )
    EDT2_gamma_6 = torch.pi - EDT2_gamma_4 - thetas[:, tids.I_THETA_EDT2_5]
    EDT2_thetatilde_4 = theta_hats[:, tids.I_THETA_EDT2_4] + EDT2_gamma_4
    EDT2_x_c6_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
        + EDT2_l_46_j_squared
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
        * EDT2_l_46_j
        * torch.cos(EDT2_thetatilde_4)
    )
    EDT2_x_c6 = torch.sqrt(EDT2_x_c6_squared)
    EDT2_phi_4_b = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        EDT2_l_46_j,
        EDT2_thetatilde_4,
    )
    EDT2_thetatilde_6 = theta_hats[:, tids.I_THETA_ALL_6] + EDT2_gamma_6
    EDT2_thetatilde_6_a = torch.pi - EDT2_thetatilde_4 - EDT2_phi_4_b
    EDT2_thetatilde_6_b = EDT2_thetatilde_6 - EDT2_thetatilde_6_a
    EDT2_l_cc_squared = (
        EDT2_x_c6_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2
        * EDT2_x_c6
        * tendon_data.link_lengths[:, tids.I_LINK_67]
        * torch.cos(EDT2_thetatilde_6_b)
    )
    EDT2_l_cc_C = torch.sqrt(EDT2_l_cc_squared)
    EDT2_phi_4_d = angle_from_sws(
        EDT2_x_c6, tendon_data.link_lengths[:, tids.I_LINK_67], EDT2_thetatilde_6_b
    )

    EDT2_h6_C = EDT2_x_c6 * torch.sin(EDT2_phi_4_d)
    EDT2_h5_C = EDT2_h6_C - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.sin(
        EDT2_gamma_6
    )
    EDT2_L_s_C = EDT2_l_cc_C

    # state D: tendon wraps around j5 pulley but not j6 pulley
    EDT2_x_56_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * tendon_data.link_lengths[:, tids.I_LINK_67]
        * torch.cos(theta_hats[:, tids.I_THETA_ALL_6])
    )
    EDT2_x_56 = torch.sqrt(EDT2_x_56_squared)
    EDT2_l_5c_D = torch.sqrt(
        EDT2_x_56_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5]
    )
    EDT2_phi_56_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_67],
        theta_hats[:, tids.I_THETA_ALL_6],
    )
    EDT2_phi_56_b = torch.atan2(
        EDT2_l_5c_D, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
    )
    EDT2_phi_56 = EDT2_phi_56_a + EDT2_phi_56_b
    EDT2_q5_D = (
        2 * torch.pi
        - EDT2_phi_56
        - thetas[:, tids.I_THETA_EDT2_5]
        - EDT2_thetahat_5_a
        - EDT2_phi_45_A  # note: phi_45 is the same for states A and D
    )
    EDT2_phi_7_D = 1.5 * torch.pi - theta_hats[:, tids.I_THETA_ALL_6] - EDT2_phi_56
    EDT2_h6_D = tendon_data.link_lengths[:, tids.I_LINK_67] * torch.sin(EDT2_phi_7_D)
    EDT2_L_s_D = (
        EDT2_l_c5_A
        + EDT2_q5_D * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
        + EDT2_l_5c_D
    )

    # state decision logic
    EDT2_h5_B_disengaged = torch.where(
        EDT2_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5],
        True,
        False,
    )
    EDT2_h5_C_disengaged = torch.where(
        EDT2_h5_C > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5],
        True,
        False,
    )
    EDT2_h6_C_disengaged = torch.where(
        EDT2_h6_C > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6],
        True,
        False,
    )

    EDT2_h6_D_disengaged = torch.where(
        EDT2_h6_D > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6],
        True,
        False,
    )
    EDT2_state_C = (EDT2_h5_B_disengaged & EDT2_h6_C_disengaged) | (
        EDT2_h6_D_disengaged & EDT2_h5_C_disengaged
    )
    EDT2_state_B = ~EDT2_state_C & EDT2_h5_B_disengaged
    EDT2_state_D = ~EDT2_state_C & EDT2_h6_D_disengaged
    EDT2_state_A = ~(EDT2_state_B | EDT2_state_C | EDT2_state_D)
    EDT2_state_A = EDT2_state_A.to(torch.bool)
    EDT2_state_B = EDT2_state_B.to(torch.bool)
    EDT2_state_C = EDT2_state_C.to(torch.bool)
    EDT2_state_D = EDT2_state_D.to(torch.bool)

    EDT2_L_s = torch.where(
        EDT2_state_A,
        EDT2_L_s_A,
        torch.where(
            EDT2_state_B,
            EDT2_L_s_B,
            torch.where(EDT2_state_C, EDT2_L_s_C, EDT2_L_s_D),
        ),
    )
    EDT2_delta_L_s = tendon_data.edt2_length - EDT2_L_s

    return (
        GST_delta_L_s,
        DFT_delta_L_s,
        KFT_delta_L_s,
        EDT1_delta_L_s,
        EDT2_delta_L_s,
    )


# Running pipeline:
# 0) transform joint angles to thetas and qs
# 1) evaluate conditions
# 2) compute energy with conditional function
# 3) differentiate w.r.t. joint angles
# 4) apply torques


class TendonManager:
    # affects joints j3, j4, j5, j6 through links s23, s34, s45, s56, s67
    def __init__(
        self,
        robot: Articulation,
        tendon_data: TendonData = TendonData(1, dummy_randomization),
    ):
        self.robot = robot
        self.device = robot.device

        self.link_indices_left_right, _ = self.robot.find_bodies(
            link_names_left + link_names_right, preserve_order=True
        )
        self.joint_indices_left, _ = self.robot.find_joints(
            joint_names_left, preserve_order=True
        )
        self.joint_indices_right, _ = self.robot.find_joints(
            joint_names_right, preserve_order=True
        )

        self.hip_joint_names = [
            "l2_pseudo_acetabulofemoral_flexion",
            "r2_pseudo_acetabulofemoral_flexion",
        ]
        self.hip_static_joint_names = [
            "r0_acetabulofemoral_roll",
            "r1_acetabulofemoral_lateral",
            "l0_acetabulofemoral_roll",
            "l1_acetabulofemoral_lateral",
        ]
        self.hip_joint_indices, _ = self.robot.find_joints(
            self.hip_joint_names, preserve_order=True
        )
        self.hip_static_joint_indices, _ = self.robot.find_joints(
            self.hip_static_joint_names, preserve_order=True
        )

        self.foot_link_names = [
            link_names_left[tids.I_CHAIN_LINK_67],
            link_names_right[tids.I_CHAIN_LINK_67],
        ]
        self.foot_link_indices, _ = self.robot.find_bodies(
            self.foot_link_names, preserve_order=True
        )

        # TODO: add explanation in params to discuss these indices definitions
        self.tendon_data = tendon_data
        self.tendon_data_jit = tendon_data.to_jit()

    def compute_delta_l_s_debug(
        self, joint_angles: torch.Tensor, tendon_data: TendonData
    ):
        # 0) transform joint angles to thetas and qs
        joint_angles_signed = tendon_data.joint_directions * joint_angles
        thetas = torch.empty_like(tendon_data.tendon_offsets_theta)
        thetas[
            (
                tids.I_THETA_GST_3,
                tids.I_THETA_GST_4,
                tids.I_THETA_GST_5,
                tids.I_THETA_ALL_6,
                tids.I_THETA_DFT_5,
                tids.I_THETA_EDT1_4,
                tids.I_THETA_EDT1_5,
                tids.I_THETA_EDT2_4,
                tids.I_THETA_EDT2_5,
                tids.I_THETA_KFT_3,
                tids.I_THETA_KFT_8,
            )
        ] = (
            joint_angles_signed[
                (
                    tids.I_JOINT_3,
                    tids.I_JOINT_4,
                    tids.I_JOINT_5,
                    tids.I_JOINT_6,
                    tids.I_JOINT_5,
                    tids.I_JOINT_4,
                    tids.I_JOINT_5,
                    tids.I_JOINT_4,
                    tids.I_JOINT_5,
                    tids.I_JOINT_3,
                    tids.I_JOINT_8,
                )
            ]
            + tendon_data.tendon_offsets_theta[
                (
                    tids.I_THETA_GST_3,
                    tids.I_THETA_GST_4,
                    tids.I_THETA_GST_5,
                    tids.I_THETA_ALL_6,
                    tids.I_THETA_DFT_5,
                    tids.I_THETA_EDT1_4,
                    tids.I_THETA_EDT1_5,
                    tids.I_THETA_EDT2_4,
                    tids.I_THETA_EDT2_5,
                    tids.I_THETA_KFT_3,
                    tids.I_THETA_KFT_8,
                )
            ]
        )
        qs = torch.empty_like(tendon_data.tendon_offsets_q_theta)
        qs[
            (
                tids.I_Q_GST_3,
                tids.I_Q_GST_4,
                tids.I_Q_GST_5,
                tids.I_Q_GST_6,
                tids.I_Q_DFT_5,
                tids.I_Q_DFT_6,
            )
        ] = (
            thetas[
                (
                    tids.I_THETA_GST_3,
                    tids.I_THETA_GST_4,
                    tids.I_THETA_GST_5,
                    tids.I_THETA_ALL_6,
                    tids.I_THETA_DFT_5,
                    tids.I_THETA_ALL_6,
                )
            ]
            + tendon_data.tendon_offsets_q_theta[
                (
                    tids.I_Q_GST_3,
                    tids.I_Q_GST_4,
                    tids.I_Q_GST_5,
                    tids.I_Q_GST_6,
                    tids.I_Q_DFT_5,
                    tids.I_Q_DFT_6,
                )
            ]
        )

        theta_hats = -thetas + 2 * torch.pi
        qhats = torch.empty_like(tendon_data.tendon_offsets_qhat_thetahat)
        qhats[(tids.I_QHAT_EDT2_6,)] = (
            theta_hats[(tids.I_THETA_ALL_6,)]
            + tendon_data.tendon_offsets_qhat_thetahat[[(tids.I_QHAT_EDT2_6,)]]
        )

        ### --------------- GST --------------- ###
        # 1) evaluate conditions
        # 1a) compute h5^B
        GST_x_4prime6_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * torch.cos(thetas[:, tids.I_THETA_GST_5])
        )
        GST_x_4prime6 = torch.sqrt(GST_x_4prime6_squared)
        GST_l_4prime6_squared = GST_x_4prime6_squared - (
            (
                tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime]
                - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
            )
            ** 2
        )
        GST_l_4prime6 = torch.sqrt(GST_l_4prime6_squared)
        GST_phi_4prime_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            tendon_data.link_lengths[:, tids.I_LINK_56],
            thetas[:, tids.I_THETA_GST_5],
        )

        GST_phi_4prime_b = torch.acos(
            (
                tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_4prime]
                + GST_x_4prime6_squared
                - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_6]
                - GST_l_4prime6_squared
            )
            / (
                2
                * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime]
                * GST_x_4prime6
            )
        )
        GST_phi_4prime_B = GST_phi_4prime_a + GST_phi_4prime_b
        GST_h5_B = tendon_data.pulley_radii[
            :, tids.I_RADIUS_GST_4prime
        ] - tendon_data.link_lengths[:, tids.I_LINK_4prime5] * torch.cos(
            GST_phi_4prime_B
        )

        # 1b) compute h5^C and h6^C
        GST_theta_6_a = torch.pi - thetas[:, tids.I_THETA_GST_5] - GST_phi_4prime_a
        GST_theta_6_b = thetas[:, tids.I_THETA_ALL_6] - GST_theta_6_a
        GST_x_4prime7_squared = (
            GST_x_4prime6_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * GST_x_4prime6
            * tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(GST_theta_6_b)
        )
        GST_x_4prime7 = torch.sqrt(GST_x_4prime7_squared)
        GST_phi_4prime_d = angle_from_sws(
            GST_x_4prime6,
            tendon_data.link_lengths[:, tids.I_LINK_67],
            GST_theta_6_b,
        )

        GST_phi_4prime_c = torch.acos(
            tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] / GST_x_4prime7
        )
        GST_phi_4prime_C = GST_phi_4prime_a + GST_phi_4prime_c + GST_phi_4prime_d
        GST_h5_C = tendon_data.pulley_radii[
            :, tids.I_RADIUS_GST_4prime
        ] - tendon_data.link_lengths[:, tids.I_LINK_4prime5] * torch.cos(
            GST_phi_4prime_C
        )
        GST_h6_C = tendon_data.pulley_radii[
            :, tids.I_RADIUS_GST_4prime
        ] - GST_x_4prime6 * torch.cos(GST_phi_4prime_c + GST_phi_4prime_d)

        # print("Theta 6:", thetas[:, self.JOINT_ANGLES_6])

        # 1c) compute h6^D
        GST_x_57_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(
                thetas[:, tids.I_THETA_ALL_6],
            )
        )
        GST_x_57 = torch.sqrt(GST_x_57_squared)
        GST_l_57_squared = (
            GST_x_57_squared - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] ** 2
        )
        GST_l_57 = torch.sqrt(GST_l_57_squared)

        GST_phi_5_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_56],
            tendon_data.link_lengths[:, tids.I_LINK_67],
            thetas[:, tids.I_THETA_ALL_6],
        )

        GST_phi_5_b = torch.acos(
            tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] / GST_x_57
        )
        GST_phi_5_D = GST_phi_5_a + GST_phi_5_b
        GST_h6_D = tendon_data.pulley_radii[
            :, tids.I_RADIUS_GST_5
        ] - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.cos(GST_phi_5_D)

        GST_h5_B_disengaged = torch.where(
            GST_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5],
            True,
            False,
        )
        GST_h5_C_disengaged = torch.where(
            GST_h5_C > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5],
            True,
            False,
        )
        GST_h6_C_disengaged = torch.where(
            GST_h6_C > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6],
            True,
            False,
        )

        GST_h6_D_disengaged = torch.where(
            GST_h6_D > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6],
            True,
            False,
        )
        GST_state_C = (GST_h5_B_disengaged & GST_h6_C_disengaged) | (
            GST_h6_D_disengaged & GST_h5_C_disengaged
        )
        GST_state_B = ~GST_state_C & GST_h5_B_disengaged
        GST_state_D = ~GST_state_C & GST_h6_D_disengaged
        assert (
            GST_state_B.sum() + GST_state_D.sum() == (GST_state_B | GST_state_D).sum()
        ), "States B and D are active simultaneously"
        GST_state_A = ~(GST_state_B | GST_state_C | GST_state_D)

        # 2) compute energy with conditional function for lower tendon state length
        # state A
        GST_lower_tendon_state_length_after_4prime_A = (
            tendon_data.tendon_section_lengths[
                :, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5
            ]
            + qs[:, tids.I_Q_GST_5] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
            + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_56]
            + qs[:, tids.I_Q_GST_6] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
            + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_67]
        )

        # state B
        GST_q6_B = (
            thetas[:, tids.I_THETA_ALL_6]
            - tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENCY_ANGLE_GST_67_J6
            ]
            - 2 * torch.pi
            + GST_phi_4prime_B
            + thetas[:, tids.I_THETA_GST_5]
        )

        GST_lower_tendon_state_length_after_4prime_B = (
            GST_l_4prime6
            + GST_q6_B * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
            + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_67]
        )

        # state C
        GST_l_4prime7_squared = (
            GST_x_4prime7_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_4prime]
        )
        GST_l_4prime7 = torch.sqrt(GST_l_4prime7_squared)
        GST_lower_tendon_state_length_after_4prime_C = GST_l_4prime7

        # state D
        GST_q5_D = (
            thetas[:, tids.I_THETA_GST_5]
            - tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J5
            ]
            - GST_phi_5_D
        )
        GST_lower_tendon_state_length_after_4prime_D = (
            tendon_data.tendon_section_lengths[
                :, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5
            ]
            + GST_q5_D * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
            + GST_l_57
        )

        GST_lower_tendon_state_length_after_4prime = torch.where(
            GST_state_A,
            GST_lower_tendon_state_length_after_4prime_A,
            torch.where(
                GST_state_B,
                GST_lower_tendon_state_length_after_4prime_B,
                torch.where(
                    GST_state_C,
                    GST_lower_tendon_state_length_after_4prime_C,
                    GST_lower_tendon_state_length_after_4prime_D,
                ),
            ),
        )

        GST_q4prime = (
            tendon_data.lower_gst_length - GST_lower_tendon_state_length_after_4prime
        ) / tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime]

        GST_q4_base = (
            thetas[:, tids.I_THETA_GST_4]
            - GST_q4prime
            - tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENCY_ANGLE_GST_34_J4
            ]
        )

        # Use torch.where instead of in-place masked operations to preserve gradient flow
        GST_q4_adjustment = torch.where(
            torch.logical_or(GST_state_A, GST_state_D),
            tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J4
            ],
            torch.where(
                GST_state_B,
                GST_phi_4prime_B,
                GST_phi_4prime_C,  # state_C is the only remaining case
            ),
        )
        GST_q4 = GST_q4_base - GST_q4_adjustment

        GST_delta_L_s = (
            tendon_data.upper_gst_length
            - tendon_data.gst_spring_rest_length
            - tendon_data.link_lengths[:, tids.I_LINK_GST_23]
            - qs[:, tids.I_Q_GST_3] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_3]
            - tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_34]
            - GST_q4 * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4]
        )

        ### --------- DFT ---------- ###
        DFT_q5 = qs[:, tids.I_Q_DFT_5]
        DFT_q6 = qs[:, tids.I_Q_DFT_6]
        DFT_l_c5 = tendon_data.tendon_section_lengths[
            :, tids.I_TENDON_SECTION_LENGTH_DFT_C5
        ]
        DFT_l_56 = tendon_data.tendon_section_lengths[
            :, tids.I_TENDON_SECTION_LENGTH_DFT_56
        ]
        DFT_l_6c = tendon_data.tendon_section_lengths[
            :, tids.I_TENDON_SECTION_LENGTH_DFT_6C
        ]

        DFT_delta_L_s = (
            tendon_data.dft_length
            - DFT_l_c5
            - DFT_q5 * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5]
            - DFT_l_56
            - DFT_q6 * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6]
            - DFT_l_6c
        )

        ### --------- KFT ---------- ###
        KFT_l_8c_j_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_83]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_KFT_3C]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_83]
            * tendon_data.link_lengths[:, tids.I_LINK_KFT_3C]
            * torch.cos(theta_hats[:, tids.I_THETA_KFT_3])
        )
        KFT_l_8c = torch.sqrt(
            KFT_l_8c_j_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_KFT_8]
        )
        KFT_phi_8 = torch.atan2(
            KFT_l_8c, tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8]
        )
        KFT_phi_8_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_83],
            tendon_data.link_lengths[:, tids.I_LINK_KFT_3C],
            theta_hats[:, tids.I_THETA_KFT_3],
        )
        KFT_q8 = thetas[:, tids.I_THETA_KFT_8] - KFT_phi_8 + KFT_phi_8_a

        KFT_delta_L_s = (
            tendon_data.kft_length
            - KFT_q8 * tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8]
            - KFT_l_8c
        )

        ### ------------- EDT1 ------------- ###

        EDT1_x_c5_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_C4]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4]
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * torch.cos(theta_hats[:, tids.I_THETA_EDT1_4])
        )
        EDT1_x_c5 = torch.sqrt(EDT1_x_c5_squared)
        EDT1_phi_4_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4],
            tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            theta_hats[:, tids.I_THETA_EDT1_4],
        )
        EDT1_thetahat_5_a = torch.pi - theta_hats[:, tids.I_THETA_EDT1_4] - EDT1_phi_4_a

        # state A: tendon wraps around j5 pulley
        EDT1_l_c5_A = torch.sqrt(
            EDT1_x_c5_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT1_5]
        )
        EDT1_phi_45_A = torch.atan2(
            EDT1_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
        )
        EDT1_q5_A = (
            2 * torch.pi
            - tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENCY_ANGLE_EDT1_5C_J5
            ]
            - thetas[:, tids.I_THETA_EDT1_5]
            - EDT1_thetahat_5_a
            - EDT1_phi_45_A
        )

        # state B: tendon does not wrap around j5 pulley

        EDT1_thetahat_5_b = theta_hats[:, tids.I_THETA_EDT1_5] - EDT1_thetahat_5_a
        EDT1_phi_4_b = angle_from_sws(
            EDT1_x_c5,
            tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C],
            EDT1_thetahat_5_b,
        )

        EDT1_h5_B = EDT1_x_c5 * torch.sin(EDT1_phi_4_b)
        EDT1_l_cc = torch.sqrt(
            EDT1_x_c5_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_5C]
            - 2
            * EDT1_x_c5
            * tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C]
            * torch.cos(EDT1_thetahat_5_b)
        )

        EDT1_state_B = EDT1_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
        EDT1_state_A = ~EDT1_state_B

        EDT1_L_s = torch.where(
            EDT1_state_B,
            EDT1_l_cc,
            EDT1_l_c5_A
            + EDT1_q5_A * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
            + tendon_data.tendon_section_lengths[
                :, tids.I_TENDON_SECTION_LENGTH_EDT1_5C
            ],
        )

        EDT1_delta_L_s = tendon_data.edt1_length - EDT1_L_s

        ### ------------- EDT2 ------------- ###
        EDT2_x_c5_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * torch.cos(theta_hats[:, tids.I_THETA_EDT2_4])
        )
        EDT2_x_c5 = torch.sqrt(EDT2_x_c5_squared)
        EDT2_phi_4_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
            tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            theta_hats[:, tids.I_THETA_EDT2_4],
        )
        EDT2_thetahat_5_a = torch.pi - theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_phi_4_a

        # state A: tendon wraps around j5 and j6 pulleys
        EDT2_l_c5_A = torch.sqrt(
            EDT2_x_c5_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5]
        )
        EDT2_phi_45_A = torch.atan2(
            EDT2_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
        )
        EDT2_q5_A = (
            2 * torch.pi
            - tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENCY_ANGLE_EDT2_56_J5
            ]
            - thetas[:, tids.I_THETA_EDT2_5]
            - EDT2_thetahat_5_a
            - EDT2_phi_45_A
        )
        EDT2_L_s_A = (
            EDT2_l_c5_A
            + EDT2_q5_A * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
            + tendon_data.tendon_section_lengths[
                :, tids.I_TENDON_SECTION_LENGTH_EDT2_56
            ]
            + qhats[:, tids.I_QHAT_EDT2_6]
            * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
            + tendon_data.tendon_section_lengths[
                :, tids.I_TENDON_SECTION_LENGTH_EDT2_6C
            ]
        )

        # state B: tendon wraps around j6 pulley but not j5 pulley
        EDT2_x_64prime_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * torch.cos(thetas[:, tids.I_THETA_EDT2_5])
        )
        EDT2_x_64prime = torch.sqrt(EDT2_x_64prime_squared)
        EDT2_phi_6_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_56],
            tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            theta_hats[:, tids.I_THETA_EDT2_5],
        )
        EDT2_thetahat_4_a = torch.pi - theta_hats[:, tids.I_THETA_EDT2_5] - EDT2_phi_6_a
        EDT2_thetahat_4_b = theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_thetahat_4_a
        EDT2_x_6c_squared = (
            EDT2_x_64prime_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
            - 2
            * EDT2_x_64prime
            * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
            * torch.cos(EDT2_thetahat_4_b)
        )
        EDT2_x_6c = torch.sqrt(EDT2_x_6c_squared)
        EDT2_phi_6_d = angle_from_sws(
            EDT2_x_6c,
            tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
            EDT2_thetahat_4_b,
        )
        EDT2_l_c6_B = torch.sqrt(
            EDT2_x_6c_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_6]
        )
        EDT2_phi_6_c = torch.atan2(
            EDT2_l_c6_B, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
        )
        EDT2_phi_6_B = EDT2_phi_6_a + EDT2_phi_6_c + EDT2_phi_6_d
        EDT2_q6_B = theta_hats[:, tids.I_THETA_ALL_6] - EDT2_phi_6_B
        EDT2_h5_B = tendon_data.pulley_radii[
            :, tids.I_RADIUS_EDT2_6
        ] - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.cos(EDT2_phi_6_B)
        EDT2_L_s_B = (
            EDT2_l_c6_B
            + EDT2_q6_B * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
            + tendon_data.tendon_section_lengths[
                :, tids.I_TENDON_SECTION_LENGTH_EDT2_6C
            ]
        )

        # state C: tendon does not wrap around any pulley
        EDT2_l_46_j_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * torch.cos(thetas[:, tids.I_THETA_EDT2_5])
        )
        EDT2_l_46_j = torch.sqrt(EDT2_l_46_j_squared)
        EDT2_gamma_4 = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            tendon_data.link_lengths[:, tids.I_LINK_56],
            thetas[:, tids.I_THETA_EDT2_5],
        )
        EDT2_gamma_6 = torch.pi - EDT2_gamma_4 - thetas[:, tids.I_THETA_EDT2_5]
        EDT2_thetatilde_4 = theta_hats[:, tids.I_THETA_EDT2_4] + EDT2_gamma_4
        EDT2_x_c6_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
            + EDT2_l_46_j_squared
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
            * EDT2_l_46_j
            * torch.cos(EDT2_thetatilde_4)
        )
        EDT2_x_c6 = torch.sqrt(EDT2_x_c6_squared)
        EDT2_phi_4_b = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
            EDT2_l_46_j,
            EDT2_thetatilde_4,
        )
        EDT2_thetatilde_6 = theta_hats[:, tids.I_THETA_ALL_6] + EDT2_gamma_6
        EDT2_thetatilde_6_a = torch.pi - EDT2_thetatilde_4 - EDT2_phi_4_b
        EDT2_thetatilde_6_b = EDT2_thetatilde_6 - EDT2_thetatilde_6_a
        EDT2_l_cc_squared = (
            EDT2_x_c6_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * EDT2_x_c6
            * tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(EDT2_thetatilde_6_b)
        )
        EDT2_l_cc_C = torch.sqrt(EDT2_l_cc_squared)
        EDT2_phi_4_d = angle_from_sws(
            EDT2_x_c6, tendon_data.link_lengths[:, tids.I_LINK_67], EDT2_thetatilde_6_b
        )

        EDT2_h6_C = EDT2_x_c6 * torch.sin(EDT2_phi_4_d)
        EDT2_h5_C = EDT2_h6_C - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.sin(
            EDT2_gamma_6
        )
        EDT2_L_s_C = EDT2_l_cc_C

        # state D: tendon wraps around j5 pulley but not j6 pulley
        EDT2_x_56_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(theta_hats[:, tids.I_THETA_ALL_6])
        )
        EDT2_x_56 = torch.sqrt(EDT2_x_56_squared)
        EDT2_l_5c_D = torch.sqrt(
            EDT2_x_56_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5]
        )
        EDT2_phi_56_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_56],
            tendon_data.link_lengths[:, tids.I_LINK_67],
            theta_hats[:, tids.I_THETA_ALL_6],
        )
        EDT2_phi_56_b = torch.atan2(
            EDT2_l_5c_D, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
        )
        EDT2_phi_56 = EDT2_phi_56_a + EDT2_phi_56_b
        EDT2_q5_D = (
            2 * torch.pi
            - EDT2_phi_56
            - thetas[:, tids.I_THETA_EDT2_5]
            - EDT2_thetahat_5_a
            - EDT2_phi_45_A  # note: phi_45 is the same for states A and D
        )
        EDT2_phi_7_D = 1.5 * torch.pi - theta_hats[:, tids.I_THETA_ALL_6] - EDT2_phi_56
        EDT2_h6_D = tendon_data.link_lengths[:, tids.I_LINK_67] * torch.sin(
            EDT2_phi_7_D
        )
        EDT2_L_s_D = (
            EDT2_l_c5_A
            + EDT2_q5_D * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
            + EDT2_l_5c_D
        )

        # state decision logic
        EDT2_h5_B_disengaged = torch.where(
            EDT2_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5],
            True,
            False,
        )
        EDT2_h5_C_disengaged = torch.where(
            EDT2_h5_C > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5],
            True,
            False,
        )
        EDT2_h6_C_disengaged = torch.where(
            EDT2_h6_C > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6],
            True,
            False,
        )

        EDT2_h6_D_disengaged = torch.where(
            EDT2_h6_D > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6],
            True,
            False,
        )
        EDT2_state_C = (EDT2_h5_B_disengaged & EDT2_h6_C_disengaged) | (
            EDT2_h6_D_disengaged & EDT2_h5_C_disengaged
        )
        EDT2_state_B = ~EDT2_state_C & EDT2_h5_B_disengaged
        EDT2_state_D = ~EDT2_state_C & EDT2_h6_D_disengaged
        EDT2_state_A = ~(EDT2_state_B | EDT2_state_C | EDT2_state_D)
        EDT2_state_A = EDT2_state_A.to(torch.bool)
        EDT2_state_B = EDT2_state_B.to(torch.bool)
        EDT2_state_C = EDT2_state_C.to(torch.bool)
        EDT2_state_D = EDT2_state_D.to(torch.bool)

        EDT2_L_s = torch.where(
            EDT2_state_A,
            EDT2_L_s_A,
            torch.where(
                EDT2_state_B,
                EDT2_L_s_B,
                torch.where(EDT2_state_C, EDT2_L_s_C, EDT2_L_s_D),
            ),
        )
        EDT2_delta_L_s = tendon_data.edt2_length - EDT2_L_s

        return (
            GST_delta_L_s,
            DFT_delta_L_s,
            KFT_delta_L_s,
            EDT1_delta_L_s,
            EDT2_delta_L_s,
            {
                "thetas": thetas,
                "qs": qs,
                "qhats": qhats,
                "GST_state_a": GST_state_A,
                "GST_state_b": GST_state_B,
                "GST_state_c": GST_state_C,
                "GST_state_d": GST_state_D,
                "EDT1_state_a": EDT1_state_A,
                "EDT1_state_b": EDT1_state_B,
                "EDT2_state_a": EDT2_state_A,
                "EDT2_state_b": EDT2_state_B,
                "EDT2_state_c": EDT2_state_C,
                "EDT2_state_d": EDT2_state_D,
                "GST_delta_L_s": GST_delta_L_s,
                "DFT_delta_L_s": DFT_delta_L_s,
                "KFT_delta_L_s": KFT_delta_L_s,
                "EDT1_delta_L_s": EDT1_delta_L_s,
                "EDT2_delta_L_s": EDT2_delta_L_s,
                "GST_q4": GST_q4,
                "GST_q4prime": GST_q4prime,
                "GST_q5_D": GST_q5_D,
                "GST_q6_B": GST_q6_B,
                "GST_l_4prime6": GST_l_4prime6,
                "GST_l_4prime7": GST_l_4prime7,
                "GST_l_57": GST_l_57,
                "GST_x_4prime6": GST_x_4prime6,
                "GST_x_4prime7": GST_x_4prime7,
                "GST_x_57": GST_x_57,
                "GST_phi_4prime_a": GST_phi_4prime_a,
                "GST_phi_4prime_b": GST_phi_4prime_b,
                "GST_phi_4prime_c": GST_phi_4prime_c,
                "GST_phi_4prime_d": GST_phi_4prime_d,
                "GST_phi_5_a": GST_phi_5_a,
                "GST_phi_5_b": GST_phi_5_b,
                "GST_h5_B": GST_h5_B,
                "GST_h5_C": GST_h5_C,
                "GST_h6_C": GST_h6_C,
                "GST_h6_D": GST_h6_D,
                "KFT_l_8c": KFT_l_8c,
                "KFT_phi_8": KFT_phi_8,
                "KFT_phi_8_a": KFT_phi_8_a,
                "KFT_q8": KFT_q8,
                "EDT1_x_c5": EDT1_x_c5,
                "EDT1_x_c5": EDT1_x_c5,
                "EDT1_phi_4_a": EDT1_phi_4_a,
                "EDT1_thetahat_5_a": EDT1_thetahat_5_a,
                "EDT1_l_c5_A": EDT1_l_c5_A,
                "EDT1_phi_45_A": EDT1_phi_45_A,
                "EDT1_q5_A": EDT1_q5_A,
                "EDT1_thetahat_5_b": EDT1_thetahat_5_b,
                "EDT1_phi_4_b": EDT1_phi_4_b,
                "EDT1_h5_B": EDT1_h5_B,
                "EDT1_l_cc": EDT1_l_cc,
                "EDT2_x_c5": EDT2_x_c5,
                "EDT2_phi_4_a": EDT2_phi_4_a,
                "EDT2_thetahat_5_a": EDT2_thetahat_5_a,
                "EDT2_l_c5_A": EDT2_l_c5_A,
                "EDT2_phi_45_A": EDT2_phi_45_A,
                "EDT2_q5_A": EDT2_q5_A,
                "EDT2_x_64prime": EDT2_x_64prime,
                "EDT2_phi_6_a": EDT2_phi_6_a,
                "EDT2_thetahat_4_a": EDT2_thetahat_4_a,
                "EDT2_thetahat_4_b": EDT2_thetahat_4_b,
                "EDT2_x_6c": EDT2_x_6c,
                "EDT2_phi_6_d": EDT2_phi_6_d,
                "EDT2_l_c6_B": EDT2_l_c6_B,
                "EDT2_phi_6_c": EDT2_phi_6_c,
                "EDT2_phi_6_B": EDT2_phi_6_B,
                "EDT2_q6_B": EDT2_q6_B,
                "EDT2_h5_B": EDT2_h5_B,
                "EDT2_l_46_j": EDT2_l_46_j,
                "EDT2_gamma_4": EDT2_gamma_4,
                "EDT2_gamma_6": EDT2_gamma_6,
                "EDT2_thetatilde_4": EDT2_thetatilde_4,
                "EDT2_x_c6": EDT2_x_c6,
                "EDT2_phi_4_b": EDT2_phi_4_b,
                "EDT2_thetatilde_6": EDT2_thetatilde_6,
                "EDT2_thetatilde_6_a": EDT2_thetatilde_6_a,
                "EDT2_thetatilde_6_b": EDT2_thetatilde_6_b,
                "EDT2_l_cc_C": EDT2_l_cc_C,
                "EDT2_phi_4_d": EDT2_phi_4_d,
                "EDT2_h6_C": EDT2_h6_C,
                "EDT2_h5_C": EDT2_h5_C,
                "EDT2_x_56": EDT2_x_56,
                "EDT2_l_5c_D": EDT2_l_5c_D,
                "EDT2_phi_56_a": EDT2_phi_56_a,
                "EDT2_phi_56_b": EDT2_phi_56_b,
                "EDT2_phi_56": EDT2_phi_56,
                "EDT2_q5_D": EDT2_q5_D,
                "EDT2_phi_7_D": EDT2_phi_7_D,
                "EDT2_h6_D": EDT2_h6_D,
            },
        )

    def compute_torques_debug(self):
        batch_size = self.robot.num_instances
        joint_angles = torch.cat(
            (
                self.robot.data.joint_pos[:, self.joint_indices_left]
                .clone()
                .requires_grad_(True),
                self.robot.data.joint_pos[:, self.joint_indices_right]
                .clone()
                .requires_grad_(True),
            ),
            dim=0,
        )  # q3, q4, q5, q6, (2*N_envs) x 4 joints

        (
            GST_delta_L_s,
            DFT_delta_L_s,
            KFT_delta_L_s,
            EDT1_delta_L_s,
            EDT2_delta_L_s,
            info,
        ) = self.compute_delta_l_s_debug(joint_angles, self.tendon_data)

        GST_not_slack = GST_delta_L_s <= 0.0
        DFT_not_slack = DFT_delta_L_s <= 0.0
        KFT_not_slack = KFT_delta_L_s <= 0.0
        EDT1_not_slack = EDT1_delta_L_s <= 0.0
        EDT2_not_slack = EDT2_delta_L_s <= 0.0

        GST_energy = 0.5 * self.tendon_data.gst_stiffness * GST_delta_L_s**2
        DFT_energy = 0.5 * self.tendon_data.dft_stiffness * DFT_delta_L_s**2
        KFT_energy = 0.5 * self.tendon_data.kft_stiffness * KFT_delta_L_s**2
        EDT1_energy = 0.5 * self.tendon_data.edt1_stiffness * EDT1_delta_L_s**2
        EDT2_energy = 0.5 * self.tendon_data.edt2_stiffness * EDT2_delta_L_s**2

        total_energy = (
            GST_energy[GST_not_slack].sum()
            + DFT_energy[DFT_not_slack].sum()
            + KFT_energy[KFT_not_slack].sum()
            + EDT1_energy[EDT1_not_slack].sum()
            + EDT2_energy[EDT2_not_slack].sum()
        )

        tendon_torques = torch.autograd.grad(
            outputs=total_energy,
            inputs=joint_angles,
            create_graph=False,
            allow_unused=True,
        )[0]
        tendon_torques_left = tendon_torques[:batch_size]
        tendon_torques_right = tendon_torques[batch_size:]

        info["tendon_torques_left"] = tendon_torques_left
        info["tendon_torques_right"] = tendon_torques_right

        return (tendon_torques_left, tendon_torques_right, info)

    # @torch.jit.script
    def compute_torques_jit(self):
        batch_size = self.robot.num_instances
        joint_angles = torch.cat(
            (
                self.robot.data.joint_pos[:, self.joint_indices_left]
                .clone()
                .requires_grad_(True),
                self.robot.data.joint_pos[:, self.joint_indices_right]
                .clone()
                .requires_grad_(True),
            ),
            dim=0,
        )  # q3, q4, q5, q6, (2*N_envs) x 4 joints

        delta_L_s = compute_delta_l_s_jit(joint_angles, self.tendon_data_jit)

        not_slack = delta_L_s <= 0.0

        energy = 0.5 * self.tendon_data.gst_stiffness * delta_L_s**2

        tendon_torques = torch.autograd.grad(
            outputs=energy[not_slack].sum(),
            inputs=joint_angles,
            create_graph=False,
            allow_unused=True,
        )[0]
        tendon_torques_left = tendon_torques[:batch_size]
        tendon_torques_right = tendon_torques[batch_size:]

        return tendon_torques_left, tendon_torques_right

    # @torch.jit.script_method
    def apply_jit(self):
        tendon_torques_left, tendon_torques_right = self.compute_torques_jit()
        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros(
            (batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device
        )
        # print("Tendon torques left shape:", tendon_torques_left.shape)
        # print("Tendon torques right shape:", tendon_torques_right.shape)

        tendon_torques_full[:, : N_CHAIN_LINKS_PER_LEG - 1, 2] = -tendon_torques_left
        tendon_torques_full[:, 1:N_CHAIN_LINKS_PER_LEG, 2] += tendon_torques_left
        tendon_torques_full[
            :, N_CHAIN_LINKS_PER_LEG : N_CHAIN_LINKS_PER_LEG * 2 - 1, 2
        ] = -tendon_torques_right
        tendon_torques_full[:, N_CHAIN_LINKS_PER_LEG + 1 :, 2] += tendon_torques_right

        self.robot.set_external_force_and_torque(
            forces=torch.zeros(
                (batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device
            ),
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

        return

    def apply_debug(self):
        print("Applying GST tendon model...")

        tendon_torques_left, tendon_torques_right, info = self.compute_torques_debug()
        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros(
            (batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device
        )
        # print("Tendon torques left shape:", tendon_torques_left.shape)
        # print("Tendon torques right shape:", tendon_torques_right.shape)

        tendon_torques_full[:, : N_CHAIN_LINKS_PER_LEG - 1, 2] = -tendon_torques_left
        tendon_torques_full[:, 1:N_CHAIN_LINKS_PER_LEG, 2] += tendon_torques_left
        tendon_torques_full[
            :, N_CHAIN_LINKS_PER_LEG : N_CHAIN_LINKS_PER_LEG * 2 - 1, 2
        ] = -tendon_torques_right
        tendon_torques_full[:, N_CHAIN_LINKS_PER_LEG + 1 :, 2] += tendon_torques_right

        self.robot.set_external_force_and_torque(
            forces=torch.zeros(
                (batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device
            ),
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

        print("Applied GST tendon model.")
        return info
