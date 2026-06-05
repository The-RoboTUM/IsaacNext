# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GST tendon manager implementation."""

import torch

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.assets.articulation import Articulation
from isaaclab.tendons.models.analytic.constants import (
    JOINT_AXIS_IDX,
    dummy_randomization,
    joint_names_left,
    joint_names_right,
    link_names_left,
    link_names_right,
)
from isaaclab.tendons.models.analytic.constants import N_CHAIN_LINKS_PER_LEG as N_LINKS_PER_LEG
from isaaclab.tendons.models.analytic.tendon_data import TendonData, TendonDataJIT
from isaaclab.utils.math import quat_apply_inverse


# todo comment
@torch.jit.script
def angle_from_sws(a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    x = a - b * torch.cos(theta)
    y = b * torch.sin(theta)
    return torch.atan2(y, x)


@torch.jit.script
def compute_delta_l_s_jit(joint_angles: torch.Tensor, tendon_data: TendonDataJIT) -> torch.Tensor:
    # bad = ~torch.isfinite(joint_angles)
    # if bad.any():
    #     print("joint_angles contains non-finite values")
    #     print("joint_angles =", joint_angles)
    #     print("Bad mask =", bad)
    #     print("Bad indices =", bad.nonzero(as_tuple=False))
    #     print("Bad values before fix =", joint_angles[bad])
    #
    #     joint_angles = torch.where(bad, torch.zeros_like(joint_angles), joint_angles)
    #
    #     print("Bad values after fix =", joint_angles[bad])

    # 0) transform joint angles to thetas and qs
    joint_angles_signed = tendon_data.joint_directions * joint_angles
    thetas = joint_angles_signed + tendon_data.joint_offsets_theta
    qs = joint_angles_signed + tendon_data.joint_offsets_gst_q

    # 1) evaluate conditions
    # 1a) compute h5^B
    x_4prime6_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * torch.cos(thetas[:, tids.GST_I_Q_OFFSET_5])
    )
    x_4prime6 = torch.sqrt(x_4prime6_squared)
    l_4prime6_squared = x_4prime6_squared - (
        (tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6]) ** 2
    )
    l_4prime6 = torch.sqrt(l_4prime6_squared)
    phi_4prime_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.GST_I_Q_OFFSET_5],
    )

    phi_4prime_b = torch.acos(
        (
            tendon_data.pulley_radii_squared[:, tids.GST_I_RADIUS_4prime]
            + x_4prime6_squared
            - tendon_data.pulley_radii_squared[:, tids.GST_I_RADIUS_6]
            - l_4prime6_squared
        )
        / (2 * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] * x_4prime6)
    )
    phi_4prime_B = phi_4prime_a + phi_4prime_b
    h5_B = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - tendon_data.link_lengths[
        :, tids.I_LINK_4prime5
    ] * torch.cos(phi_4prime_B)

    # 1b) compute h5^C and h6^C
    theta_6_a = torch.pi - thetas[:, tids.GST_I_Q_OFFSET_5] - phi_4prime_a
    theta_6_b = thetas[:, tids.GST_I_Q_OFFSET_6] - theta_6_a
    x_4prime7_squared = (
        x_4prime6_squared
        + tendon_data.link_lengths_squared[:, tids.GST_I_LINK_67]
        - 2 * x_4prime6 * tendon_data.link_lengths[:, tids.GST_I_LINK_67] * torch.cos(theta_6_b)
    )
    x_4prime7 = torch.sqrt(x_4prime7_squared)
    phi_4prime_d = angle_from_sws(
        x_4prime6,
        tendon_data.link_lengths[:, tids.GST_I_LINK_67],
        theta_6_b,
    )

    phi_4prime_c = torch.acos(tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] / x_4prime7)
    phi_4prime_C = phi_4prime_a + phi_4prime_c + phi_4prime_d
    h5_C = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - tendon_data.link_lengths[
        :, tids.I_LINK_4prime5
    ] * torch.cos(phi_4prime_C)
    h6_C = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - x_4prime6 * torch.cos(phi_4prime_c + phi_4prime_d)

    # print("Theta 6:", thetas[:, self.JOINT_ANGLES_6])

    # 1c) compute h6^D
    x_57_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        + tendon_data.link_lengths_squared[:, tids.GST_I_LINK_67]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * tendon_data.link_lengths[:, tids.GST_I_LINK_67]
        * torch.cos(
            thetas[:, tids.GST_I_Q_OFFSET_6],
        )
    )
    x_57 = torch.sqrt(x_57_squared)
    l_57_squared = x_57_squared - tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5] ** 2
    l_57 = torch.sqrt(l_57_squared)

    phi_5_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.GST_I_LINK_67],
        thetas[:, tids.GST_I_Q_OFFSET_6],
    )

    phi_5_b = torch.acos(tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5] / x_57)
    phi_5_D = phi_5_a + phi_5_b
    h6_D = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5] - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.cos(
        phi_5_D
    )

    h5_B_disengaged = torch.where(
        h5_B > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5],
        True,
        False,
    )
    h5_C_disengaged = torch.where(
        h5_C > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5],
        True,
        False,
    )
    h6_C_disengaged = torch.where(
        h6_C > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6],
        True,
        False,
    )

    h6_D_disengaged = torch.where(
        h6_D > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6],
        True,
        False,
    )
    state_C = (h5_B_disengaged & h6_C_disengaged) | (h6_D_disengaged & h5_C_disengaged)
    state_B = ~state_C & h5_B_disengaged
    state_D = ~state_C & h6_D_disengaged
    state_A = ~(state_B | state_C | state_D)
    state_A = state_A.to(torch.bool)
    state_B = state_B.to(torch.bool)
    state_C = state_C.to(torch.bool)
    state_D = state_D.to(torch.bool)

    # 2) compute energy with conditional function for lower tendon state length
    # state A
    lower_tendon_state_length_after_4prime_A = (
        tendon_data.gst_tendon_section_lengths[:, tids.I_LINK_4prime5]
        + qs[:, tids.GST_I_Q_OFFSET_5] * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5]
        + tendon_data.gst_tendon_section_lengths[:, tids.I_LINK_56]
        + qs[:, tids.GST_I_Q_OFFSET_6] * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6]
        + tendon_data.gst_tendon_section_lengths[:, tids.GST_I_LINK_67]
    )

    # state B
    q6_B = (
        thetas[:, tids.GST_I_Q_OFFSET_6]
        - tendon_data.gst_tendon_tangency_angles[:, tids.GST_I_TENDON_TANGENGY_ANGLES_67_j6]
        - 2 * torch.pi
        + phi_4prime_B
        + thetas[:, tids.GST_I_Q_OFFSET_5]
    )

    lower_tendon_state_length_after_4prime_B = (
        l_4prime6
        + q6_B * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6]
        + tendon_data.gst_tendon_section_lengths[:, tids.GST_I_LINK_67]
    )

    # state C
    l_4prime7_squared = x_4prime7_squared - tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] ** 2
    l_4prime7 = torch.sqrt(l_4prime7_squared)
    lower_tendon_state_length_after_4prime_C = l_4prime7

    # state D
    q5_D = (
        thetas[:, tids.GST_I_Q_OFFSET_5]
        - tendon_data.gst_tendon_tangency_angles[:, tids.GST_I_TENDON_TANGENGY_ANGLES_45_j5]
        - phi_5_D
    )
    lower_tendon_state_length_after_4prime_D = (
        tendon_data.gst_tendon_section_lengths[:, tids.I_LINK_4prime5]
        + q5_D * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5]
        + l_57
    )
    # lower_tendon_state_length_after_4prime = (
    #     state_A.float() * lower_tendon_state_length_after_4prime_A
    #     + state_B.float() * lower_tendon_state_length_after_4prime_B
    #     + state_C.float() * lower_tendon_state_length_after_4prime_C
    #     + state_D.float() * lower_tendon_state_length_after_4prime_D
    # )
    lower_tendon_state_length_after_4prime = torch.where(
        state_A,
        lower_tendon_state_length_after_4prime_A,
        torch.where(
            state_B,
            lower_tendon_state_length_after_4prime_B,
            torch.where(
                state_C,
                lower_tendon_state_length_after_4prime_C,
                lower_tendon_state_length_after_4prime_D,
            ),
        ),
    )

    q4prime = (tendon_data.lower_gst_length - lower_tendon_state_length_after_4prime) / tendon_data.pulley_radii[
        :, tids.GST_I_RADIUS_4prime
    ]

    q4_base = (
        thetas[:, tids.GST_I_Q_OFFSET_4]
        - q4prime
        - tendon_data.gst_tendon_tangency_angles[:, tids.GST_I_TENDON_TANGENGY_ANGLES_34_j4]
    )

    # Use torch.where instead of in-place masked operations to preserve gradient flow
    q4_adjustment = torch.where(
        torch.logical_or(state_A, state_D),
        tendon_data.gst_tendon_tangency_angles[:, tids.GST_I_TENDON_TANGENGY_ANGLES_45_j4],
        torch.where(
            state_B,
            phi_4prime_B,
            phi_4prime_C,  # state_C is the only remaining case
        ),
    )
    q4 = q4_base - q4_adjustment

    delta_L_s = (
        tendon_data.upper_gst_length
        - tendon_data.gst_spring_rest_length
        - tendon_data.gst_tendon_section_lengths[:, tids.I_CONNECTOR_LINK_GST_23]
        - qs[:, tids.GST_I_Q_OFFSET_3] * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_3]
        - tendon_data.gst_tendon_section_lengths[:, tids.I_LINK_34]
        - q4 * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4]
    )

    return delta_L_s


# Running pipeline:
# 0) transform joint angles to thetas and qs
# 1) evaluate conditions
# 2) compute energy with conditional function
# 3) differentiate w.r.t. joint angles
# 4) apply torques


class GSTTendonManager:
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
        self.joint_indices_left, _ = self.robot.find_joints(joint_names_left, preserve_order=True)
        self.joint_indices_right, _ = self.robot.find_joints(joint_names_right, preserve_order=True)

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
        self.hip_joint_indices, _ = self.robot.find_joints(self.hip_joint_names, preserve_order=True)
        self.hip_static_joint_indices, _ = self.robot.find_joints(self.hip_static_joint_names, preserve_order=True)

        self.foot_link_names = [
            link_names_left[tids.I_CHAIN_LINK_67],
            link_names_right[tids.I_CHAIN_LINK_67],
        ]
        self.foot_link_indices, _ = self.robot.find_bodies(self.foot_link_names, preserve_order=True)

        # TODO: add explanation in params to discuss these indices definitions
        self.tendon_data = tendon_data
        self.tendon_data_jit = tendon_data.to_jit()

    def compute_delta_l_s_debug(self, joint_angles: torch.Tensor, tendon_data: TendonData):
        # bad = ~torch.isfinite(joint_angles)
        # if bad.any():
        #     print("joint_angles contains non-finite values")
        #     print("joint_angles =", joint_angles)
        #     print("Bad mask =", bad)
        #     print("Bad indices =", bad.nonzero(as_tuple=False))
        #     print("Bad values before fix =", joint_angles[bad])
        #
        #     joint_angles = torch.where(bad, torch.zeros_like(joint_angles), joint_angles)
        #
        #     print("Bad values after fix =", joint_angles[bad])

        # 0) transform joint angles to thetas and qs
        joint_angles_signed = tendon_data.joint_directions * joint_angles
        thetas = joint_angles_signed + tendon_data.joint_offsets_theta
        qs = thetas + tendon_data.joint_offsets_gst_q

        # 1) evaluate conditions
        # 1a) compute h5^B
        x_4prime6_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * torch.cos(thetas[:, tids.GST_I_Q_OFFSET_5])
        )
        x_4prime6 = torch.sqrt(x_4prime6_squared)
        l_4prime6_squared = x_4prime6_squared - (
            (tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6])
            ** 2
        )
        l_4prime6 = torch.sqrt(l_4prime6_squared)
        phi_4prime_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            tendon_data.link_lengths[:, tids.I_LINK_56],
            thetas[:, tids.GST_I_Q_OFFSET_5],
        )

        phi_4prime_b = torch.acos(
            (
                tendon_data.pulley_radii_squared[:, tids.GST_I_RADIUS_4prime]
                + x_4prime6_squared
                - tendon_data.pulley_radii_squared[:, tids.GST_I_RADIUS_6]
                - l_4prime6_squared
            )
            / (2 * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] * x_4prime6)
        )
        phi_4prime_B = phi_4prime_a + phi_4prime_b
        h5_B = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - tendon_data.link_lengths[
            :, tids.I_LINK_4prime5
        ] * torch.cos(phi_4prime_B)

        # print("I am in the first")

        # 1b) compute h5^C and h6^C
        theta_6_a = torch.pi - thetas[:, tids.GST_I_Q_OFFSET_5] - phi_4prime_a
        theta_6_b = thetas[:, tids.GST_I_Q_OFFSET_6] - theta_6_a
        x_4prime7_squared = (
            x_4prime6_squared
            + tendon_data.link_lengths_squared[:, tids.GST_I_LINK_67]
            - 2 * x_4prime6 * tendon_data.link_lengths[:, tids.GST_I_LINK_67] * torch.cos(theta_6_b)
        )
        x_4prime7 = torch.sqrt(x_4prime7_squared)
        phi_4prime_d = angle_from_sws(
            x_4prime6,
            tendon_data.link_lengths[:, tids.GST_I_LINK_67],
            theta_6_b,
        )

        phi_4prime_c = torch.acos(tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] / x_4prime7)
        phi_4prime_C = phi_4prime_a + phi_4prime_c + phi_4prime_d
        h5_C = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - tendon_data.link_lengths[
            :, tids.I_LINK_4prime5
        ] * torch.cos(phi_4prime_C)
        h6_C = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] - x_4prime6 * torch.cos(
            phi_4prime_c + phi_4prime_d
        )

        # print("Theta 6:", thetas[:, self.JOINT_ANGLES_6])

        # 1c) compute h6^D
        x_57_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            + tendon_data.link_lengths_squared[:, tids.GST_I_LINK_67]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * tendon_data.link_lengths[:, tids.GST_I_LINK_67]
            * torch.cos(
                thetas[:, tids.GST_I_Q_OFFSET_6],
            )
        )
        x_57 = torch.sqrt(x_57_squared)
        l_57_squared = x_57_squared - tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5] ** 2
        l_57 = torch.sqrt(l_57_squared)

        # print("I am in the later")

        phi_5_a = angle_from_sws(
            tendon_data.link_lengths[:, tids.I_LINK_56],
            tendon_data.link_lengths[:, tids.GST_I_LINK_67],
            thetas[:, tids.GST_I_Q_OFFSET_6],
        )

        phi_5_b = torch.acos(tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5] / x_57)
        phi_5_D = phi_5_a + phi_5_b
        h6_D = tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5] - tendon_data.link_lengths[
            :, tids.I_LINK_56
        ] * torch.cos(phi_5_D)

        h5_B_disengaged = torch.where(
            h5_B > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5],
            True,
            False,
        )
        h5_C_disengaged = torch.where(
            h5_C > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_5],
            True,
            False,
        )
        h6_C_disengaged = torch.where(
            h6_C > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6],
            True,
            False,
        )

        h6_D_disengaged = torch.where(
            h6_D > tendon_data.pulley_radii[:, tids.GST_I_RADIUS_6],
            True,
            False,
        )
        state_C = (h5_B_disengaged & h6_C_disengaged) | (h6_D_disengaged & h5_C_disengaged)
        state_B = ~state_C & h5_B_disengaged
        state_D = ~state_C & h6_D_disengaged
        if state_B.sum() + state_D.sum() != (state_B | state_D).sum():
            print(
                "States B and D are active simultaneously for ",
                state_B.sum() + state_D.sum() - (state_B | state_D).sum(),
                " robots.",
            )
            print(torch.nonzero(state_B * state_D))
        # assert (
        #    state_B.sum() + state_D.sum() == (state_B | state_D).sum()
        # ), "States B and D are active simultaneously"
        state_A = ~(state_B | state_C | state_D)

        # 2) compute energy with conditional function for lower tendon state length
        lower_tendon_state_length_after_4prime = torch.zeros_like(joint_angles[:, 0])
        # state A
        lower_tendon_state_length_after_4prime[state_A] = (
            tendon_data.gst_tendon_section_lengths[state_A, tids.I_LINK_4prime5]
            + qs[state_A, tids.GST_I_Q_OFFSET_5] * tendon_data.pulley_radii[state_A, tids.GST_I_RADIUS_5]
            + tendon_data.gst_tendon_section_lengths[state_A, tids.I_LINK_56]
            + qs[state_A, tids.GST_I_Q_OFFSET_6] * tendon_data.pulley_radii[state_A, tids.GST_I_RADIUS_6]
            + tendon_data.gst_tendon_section_lengths[state_A, tids.GST_I_LINK_67]
        )

        # state B
        q6_B = (
            thetas[:, tids.GST_I_Q_OFFSET_6]
            - tendon_data.gst_tendon_tangency_angles[:, tids.GST_I_TENDON_TANGENGY_ANGLES_67_j6]
            - 2 * torch.pi
            + phi_4prime_B
            + thetas[:, tids.GST_I_Q_OFFSET_5]
        )

        lower_tendon_state_length_after_4prime[state_B] = (
            l_4prime6[state_B]
            + q6_B[state_B] * tendon_data.pulley_radii[state_B, tids.GST_I_RADIUS_6]
            + tendon_data.gst_tendon_section_lengths[state_B, tids.GST_I_LINK_67]
        )

        # print("I am in the latest")

        # state C
        l_4prime7_squared = x_4prime7_squared - tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4prime] ** 2
        l_4prime7 = torch.sqrt(l_4prime7_squared)
        lower_tendon_state_length_after_4prime[state_C] = l_4prime7[state_C]

        # state D
        q5_D = (
            thetas[:, tids.GST_I_Q_OFFSET_5]
            - tendon_data.gst_tendon_tangency_angles[:, tids.GST_I_TENDON_TANGENGY_ANGLES_45_j5]
            - phi_5_D
        )
        lower_tendon_state_length_after_4prime[state_D] = (
            tendon_data.gst_tendon_section_lengths[state_D, tids.I_LINK_4prime5]
            + q5_D[state_D] * tendon_data.pulley_radii[state_D, tids.GST_I_RADIUS_5]
            + l_57[state_D]
        )

        q4prime = (tendon_data.lower_gst_length - lower_tendon_state_length_after_4prime) / tendon_data.pulley_radii[
            :, tids.GST_I_RADIUS_4prime
        ]

        q4 = (
            thetas[:, tids.GST_I_Q_OFFSET_4]
            - q4prime
            - tendon_data.gst_tendon_tangency_angles[:, tids.GST_I_TENDON_TANGENGY_ANGLES_34_j4]
        )

        # print("I am in the mid")

        state_A_or_D = state_A | state_D
        q4[state_A_or_D] -= tendon_data.gst_tendon_tangency_angles[
            state_A_or_D, tids.GST_I_TENDON_TANGENGY_ANGLES_45_j4
        ]
        q4[state_B] -= phi_4prime_B[state_B]
        q4[state_C] -= phi_4prime_C[state_C]

        delta_L_s = (
            tendon_data.upper_gst_length
            - tendon_data.gst_spring_rest_length
            - tendon_data.gst_tendon_section_lengths[:, tids.I_CONNECTOR_LINK_GST_23]
            - qs[:, tids.GST_I_Q_OFFSET_3] * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_3]
            - tendon_data.gst_tendon_section_lengths[:, tids.I_LINK_34]
            - q4 * tendon_data.pulley_radii[:, tids.GST_I_RADIUS_4]
        )

        # print("I am in the end")

        return (
            delta_L_s,
            {
                "a": state_A,
                "b": state_B,
                "c": state_C,
                "d": state_D,
                "thetas": thetas,
                "qs": qs,
                "q4": q4,
                "q4prime": q4prime,
                "q5_D": q5_D,
                "q6_B": q6_B,
                "l_4prime6": l_4prime6,
                "l_4prime7": l_4prime7,
                "l_57": l_57,
                "x_4prime6": x_4prime6,
                "x_4prime7": x_4prime7,
                "x_57": x_57,
                "phi_4prime_a": phi_4prime_a,
                "phi_4prime_b": phi_4prime_b,
                "phi_4prime_c": phi_4prime_c,
                "phi_4prime_d": phi_4prime_d,
                "phi_5_a": phi_5_a,
                "phi_5_b": phi_5_b,
                "h5_B": h5_B,
                "h5_C": h5_C,
                "h6_C": h6_C,
                "h6_D": h6_D,
            },
        )

    def compute_torques_debug(self):
        batch_size = self.robot.num_instances
        joint_angles = torch.cat(
            (
                self.robot.data.joint_pos[:, self.joint_indices_left].clone().requires_grad_(True),
                self.robot.data.joint_pos[:, self.joint_indices_right].clone().requires_grad_(True),
            ),
            dim=0,
        )  # q3, q4, q5, q6, (2*N_envs) x 4 joints

        delta_L_s, info = self.compute_delta_l_s_debug(joint_angles, self.tendon_data)

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

        info["tendon_torques_left"] = tendon_torques_left
        info["tendon_torques_right"] = tendon_torques_right
        info["delta_l"] = delta_L_s
        return (tendon_torques_left, tendon_torques_right, info)

    # @torch.jit.script
    def compute_torques_jit(self):
        batch_size = self.robot.num_instances
        torch.autograd.set_detect_anomaly(True)
        with torch.inference_mode(False):
            # with torch.enable_grad():
            joint_angles = torch.cat(
                (
                    self.robot.data.joint_pos[:, self.joint_indices_left].clone().requires_grad_(True),
                    self.robot.data.joint_pos[:, self.joint_indices_right].clone().requires_grad_(True),
                ),
                dim=0,
            ).requires_grad_(True)  # q3, q4, q5, q6, (2*N_envs) x 4 joints

            delta_L_s = compute_delta_l_s_jit(joint_angles, self.tendon_data_jit)
            # delta_L_s, info = self.compute_delta_l_s_debug(joint_angles, self.tendon_data)

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

        # print("tendon_torques_left", tendon_torques_left)
        # print("tendon_torques_right", tendon_torques_right)

        return tendon_torques_left, tendon_torques_right

    # @torch.jit.script_method
    def apply_jit(self):
        tendon_torques_left, tendon_torques_right = self.compute_torques_jit()
        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device)
        # print("Tendon torques left shape:", tendon_torques_left.shape)
        # print("Tendon torques right shape:", tendon_torques_right.shape)

        tendon_torques_full[:, : N_LINKS_PER_LEG - 1, 2] = -tendon_torques_left
        tendon_torques_full[:, 1:N_LINKS_PER_LEG, 2] += tendon_torques_left
        tendon_torques_full[:, N_LINKS_PER_LEG : N_LINKS_PER_LEG * 2 - 1, 2] = -tendon_torques_right
        tendon_torques_full[:, N_LINKS_PER_LEG + 1 :, 2] += tendon_torques_right

        self.robot.set_external_force_and_torque(
            forces=torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device),
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

        return

    def apply_debug(self):
        print("Applying GST tendon model...")

        tendon_torques_left, tendon_torques_right, info = self.compute_torques_debug()
        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device)
        # print("Tendon torques left shape:", tendon_torques_left.shape)
        # print("Tendon torques right shape:", tendon_torques_right.shape)

        tendon_torques_full[:, : N_LINKS_PER_LEG - 1, 2] = -tendon_torques_left
        tendon_torques_full[:, 1:N_LINKS_PER_LEG, 2] += tendon_torques_left
        tendon_torques_full[:, N_LINKS_PER_LEG : N_LINKS_PER_LEG * 2 - 1, 2] = -tendon_torques_right
        tendon_torques_full[:, N_LINKS_PER_LEG + 1 :, 2] += tendon_torques_right

        self.robot.set_external_force_and_torque(
            forces=torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device),
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

        print("Applied GST tendon model.")
        return info

    def apply_actuated_debug(
        self,
        hip_position: torch.Tensor,
        knee_torque: torch.Tensor,
        virtual_ground_height: float | None = None,
        apply_tendons: bool = True,
    ):
        print("Applying GST tendon model...")

        tendon_torques_left, tendon_torques_right, info = self.compute_torques_debug()
        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device)
        # print("Tendon torques left shape:", tendon_torques_left.shape)
        # print("Tendon torques right shape:", tendon_torques_right.shape)

        tendon_torques_left[:, tids.GST_I_Q_OFFSET_3] *= -1.0
        tendon_torques_right[:, tids.GST_I_Q_OFFSET_3] *= -1.0
        tendon_torques_left[:, tids.GST_I_Q_OFFSET_4] *= -1.0
        tendon_torques_right[:, tids.GST_I_Q_OFFSET_4] *= -1.0

        tendon_torques_full[:, : N_LINKS_PER_LEG - 1, JOINT_AXIS_IDX] = -tendon_torques_left
        tendon_torques_full[:, 1:N_LINKS_PER_LEG, JOINT_AXIS_IDX] += tendon_torques_left
        tendon_torques_full[:, N_LINKS_PER_LEG : N_LINKS_PER_LEG * 2 - 1, JOINT_AXIS_IDX] = -tendon_torques_right
        tendon_torques_full[:, N_LINKS_PER_LEG + 1 :, JOINT_AXIS_IDX] += tendon_torques_right

        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_CONNECTOR_LINK_GST_23],
            JOINT_AXIS_IDX,
        ] = -torch.minimum(torch.zeros(1, device=self.device), knee_torque[0])
        tendon_torques_full[:, self.link_indices_left_right[tids.I_LINK_34], JOINT_AXIS_IDX] += torch.minimum(
            torch.zeros(1, device=self.device), knee_torque[0]
        )
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_CONNECTOR_LINK_GST_23 + N_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] = -torch.minimum(torch.zeros(1, device=self.device), knee_torque[1])
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_LINK_34 + N_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] += torch.minimum(torch.zeros(1, device=self.device), knee_torque[1])

        if not apply_tendons:
            tendon_torques_full[:, :, :] = 0.0

        forces = torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device)
        # Virtual ground contact forces
        if virtual_ground_height is not None:
            # 1) get foot position in world space
            foot_heights = self.robot.data.body_com_pos_w[:, self.foot_link_indices, 2]
            # 2) compute delta to ground plane
            penetration_depths = virtual_ground_height - foot_heights
            # 3) compute force vector proportional to penetration depth
            weight = 400.0
            forces_world = penetration_depths.clamp(min=0.0).unsqueeze(-1) * torch.tensor(
                [0.0, 0.0, weight * 20], device=self.device
            )
            print("Virtual ground forces:", forces_world)
            # 4) convert force to local coordinates and apply force at foot link
            forces[:, [tids.GST_I_LINK_67, tids.GST_I_LINK_67 + N_LINKS_PER_LEG], :] = quat_apply_inverse(
                self.robot.data.body_link_quat_w[:, self.foot_link_indices],
                forces_world,
            )

        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

        self.robot.set_joint_position_target(hip_position.to(self.device), self.hip_joint_indices)
        self.robot.set_joint_position_target(
            torch.zeros((batch_size, 4), device=self.device),
            self.hip_static_joint_indices,
        )
        print("Applied GST tendon model.")
        return info

    def apply_actuated_jit(
        self,
        hip_position: torch.Tensor,
        knee_torque: torch.Tensor,
        virtual_ground_height: float | None = None,
        apply_tendons: bool = True,
    ):
        print("Applying GST tendon model...")

        tendon_torques_left, tendon_torques_right = self.compute_torques_jit()
        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device)
        # print("Tendon torques left shape:", tendon_torques_left.shape)
        # print("Tendon torques right shape:", tendon_torques_right.shape)

        tendon_torques_left[:, tids.GST_I_Q_OFFSET_3] *= -1.0
        tendon_torques_right[:, tids.GST_I_Q_OFFSET_3] *= -1.0
        tendon_torques_left[:, tids.GST_I_Q_OFFSET_4] *= -1.0
        tendon_torques_right[:, tids.GST_I_Q_OFFSET_4] *= -1.0

        tendon_torques_full[:, : N_LINKS_PER_LEG - 1, JOINT_AXIS_IDX] = -tendon_torques_left
        tendon_torques_full[:, 1:N_LINKS_PER_LEG, JOINT_AXIS_IDX] += tendon_torques_left
        tendon_torques_full[:, N_LINKS_PER_LEG : N_LINKS_PER_LEG * 2 - 1, JOINT_AXIS_IDX] = -tendon_torques_right
        tendon_torques_full[:, N_LINKS_PER_LEG + 1 :, JOINT_AXIS_IDX] += tendon_torques_right

        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_CONNECTOR_LINK_GST_23],
            JOINT_AXIS_IDX,
        ] = -torch.minimum(torch.zeros(1, device=self.device), knee_torque[0])
        tendon_torques_full[:, self.link_indices_left_right[tids.I_LINK_34], JOINT_AXIS_IDX] += torch.minimum(
            torch.zeros(1, device=self.device), knee_torque[0]
        )
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_CONNECTOR_LINK_GST_23 + N_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] = -torch.minimum(torch.zeros(1, device=self.device), knee_torque[1])
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_LINK_34 + N_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] += torch.minimum(torch.zeros(1, device=self.device), knee_torque[1])

        if not apply_tendons:
            tendon_torques_full[:, :, :] = 0.0

        forces = torch.zeros((batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device)
        # Virtual ground contact forces
        if virtual_ground_height is not None:
            # 1) get foot position in world space
            foot_heights = self.robot.data.body_com_pos_w[:, self.foot_link_indices, 2]
            # 2) compute delta to ground plane
            penetration_depths = virtual_ground_height - foot_heights
            # 3) compute force vector proportional to penetration depth
            weight = 400.0
            forces_world = penetration_depths.clamp(min=0.0).unsqueeze(-1) * torch.tensor(
                [0.0, 0.0, weight * 20], device=self.device
            )
            print("Virtual ground forces:", forces_world)
            # 4) convert force to local coordinates and apply force at foot link
            forces[:, [tids.GST_I_LINK_67, tids.GST_I_LINK_67 + N_LINKS_PER_LEG], :] = quat_apply_inverse(
                self.robot.data.body_link_quat_w[:, self.foot_link_indices],
                forces_world,
            )

        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

        self.robot.set_joint_position_target(hip_position.to(self.device), self.hip_joint_indices)
        self.robot.set_joint_position_target(
            torch.zeros((batch_size, 4), device=self.device),
            self.hip_static_joint_indices,
        )
        print("Applied GST tendon model.")
