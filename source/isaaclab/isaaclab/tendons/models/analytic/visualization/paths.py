# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tendon attachment and path-point construction for visualization.

The comments in this file are intentionally kept close to the original massive
script because many of them are useful while validating the geometry.
"""

from __future__ import annotations

import numpy as np

from isaaclab.tendons.models.analytic.visualization.context import tc, td, tids
from isaaclab.tendons.models.analytic.visualization.kinematics import rotate_by
from isaaclab.tendons.models.analytic.visualization.states import (
    get_dft_state,
    get_edt1_state,
    get_edt2_state,
    get_gst_state,
)


def compute_gst_attachment_points(alpha_2, joint_locations, data):
    [j2, j3, j4, j5, j6, _] = joint_locations
    state = get_gst_state(data)
    q3 = data["qs"][tids.I_Q_GST_3]
    q4 = data["GST_q4"]
    q4prime = data["GST_q4prime"]
    q5 = data["qs"][tids.I_Q_GST_5]
    q6 = data["qs"][tids.I_Q_GST_6]
    l4prime6 = data["GST_l_4prime6"]
    l4prime7 = data["GST_l_4prime7"]
    l57 = data["GST_l_57"]
    q6_B = data["GST_q6_B"]
    q5_D = data["GST_q5_D"]
    direction_angle = alpha_2 - np.deg2rad(8.86)  # computed manually
    starting_point = (
        rotate_by(
            alpha_2,
            rotate_by(np.deg2rad(49.5), np.array([0.059407, 0.0]))
            + np.array([0.19, 0.0]),  # compensate for c23 != link_length_23
        )
        + j2
    )
    p3_i = starting_point + rotate_by(
        direction_angle,
        np.array(
            [
                td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_GST_23].item(),
                0.0,
            ]
        ),
    )
    p3_o = rotate_by(-q3, p3_i - j3) + j3
    direction_angle -= q3
    p4_i = p3_o + rotate_by(
        direction_angle,
        np.array(
            [
                td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_GST_34].item(),
                0.0,
            ]
        ),
    )
    p4_o = (
        rotate_by(
            q4 + np.deg2rad(45),  # note: added extra angle to avoid issues with negative values
            p4_i - j4,
        )
        + j4
    )
    direction_angle += q4
    radius_4 = np.linalg.norm(p4_o - j4)
    p4prime_i = (
        rotate_by(
            np.deg2rad(-45),  # note: to compensate for extended upper tendon drawing
            (
                (p4_o - j4)
                * (
                    radius_4
                    - td.pulley_radii[0, tids.I_RADIUS_GST_4].item()
                    + td.pulley_radii[0, tids.I_RADIUS_GST_4prime].item()
                )
                / radius_4
            ),
        )
        + j4
    )
    p4prime_o = rotate_by(q4prime, p4prime_i - j4) + j4
    direction_angle += q4prime

    upper_tendon_points = [starting_point, p3_i, p3_o, p4_i, p4_o]
    upper_tendon_joints = [j3, j4]
    # upper_tendon_q_positives = [q3 >= 0, q4 >= 0, q4 >= 0]
    upper_tendon_q_positives = [q3 >= 0, q4 >= 0]  # todo: check correctness

    lower_tendon_points = [p4prime_i, p4prime_o]
    lower_tendon_joints = [j4]
    lower_tendon_q_positives = [q4prime >= 0]
    if state[-1] == "a":
        p5_i = p4prime_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5].item(),
                    0.0,
                ]
            ),
        )
        p5_o = rotate_by(q5, p5_i - j5) + j5
        direction_angle += q5
        p6_i = p5_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_GST_56].item(),
                    0.0,
                ]
            ),
        )
        p6_o = rotate_by(q6, p6_i - j6) + j6
        direction_angle += q6
        p7 = p6_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_GST_67].item(),
                    0.0,
                ]
            ),
        )
        lower_tendon_points.extend([p5_i, p5_o, p6_i, p6_o, p7])
        lower_tendon_joints.extend([j5, j6])
        lower_tendon_q_positives.extend([q5 >= 0, q6 >= 0])
    elif state[-1] == "b":
        p6_i = p4prime_o + rotate_by(direction_angle, np.array([l4prime6, 0.0]))
        p6_o = rotate_by(q6_B, p6_i - j6) + j6
        direction_angle += q6_B
        p7 = p6_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_GST_67].item(),
                    0.0,
                ]
            ),
        )
        lower_tendon_points.extend([p6_i, p6_o, p7])
        lower_tendon_joints.extend([j6])
        lower_tendon_q_positives.extend([q6_B >= 0])

    elif state[-1] == "c":
        p7 = p4prime_o + rotate_by(direction_angle, np.array([l4prime7, 0.0]))
        lower_tendon_points.extend([p7])

    elif state[-1] == "d":
        p5_i = p4prime_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5].item(),
                    0.0,
                ]
            ),
        )
        p5_o = rotate_by(q5_D, p5_i - j5) + j5
        direction_angle += q5_D
        p7 = p5_o + rotate_by(direction_angle, np.array([l57, 0.0]))
        lower_tendon_points.extend([p5_i, p5_o, p7])
        lower_tendon_joints.extend([j5])
        lower_tendon_q_positives.extend([q5_D >= 0])
    else:
        raise ValueError(f"state {state} not recognized")

    return (
        upper_tendon_points,
        upper_tendon_joints,
        upper_tendon_q_positives,
        lower_tendon_points,
        lower_tendon_joints,
        lower_tendon_q_positives,
    )


def compute_kft_points(alpha_8, joint_locations, data, r8):
    [j2, j3, j4, j5, j6, _] = joint_locations
    q8 = data["KFT_q8"]
    p8_i = (
        rotate_by(
            alpha_8,
            np.array([r8, 0.0]),
        )
        + j2
    )
    p8_o = j2 + rotate_by(
        alpha_8 - q8,
        np.array([r8, 0.0]),
    )
    direction_angle = alpha_8 - q8 - np.pi / 2
    p3_c = p8_o + rotate_by(
        direction_angle,
        np.array(
            [
                data["KFT_l_8c"],
                0.0,
            ]
        ),
    )

    tendon_points = [p8_i, p8_o, p3_c]
    tendon_joints = [j2]
    q_positives = [q8 >= 0]

    return tendon_points, tendon_joints, q_positives


def compute_dft_points(alphas, joint_locations, data, r5, r6):
    [_j2, _j3, j4, j5, j6, p7] = joint_locations
    alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_8 = alphas
    dft_state = get_dft_state(data)

    # starting point is from j5 going offset length to j4 and orthogonally
    pc5 = j5 - rotate_by(
        alpha_4,
        np.array(
            [
                tc.connector_link_lengths_longitudinal[tids.I_CONNECTOR_LINK_DFT_C5].item(),
                tc.connector_link_lengths_lateral[tids.I_CONNECTOR_LINK_DFT_C5].item(),
            ]
        ),
    )
    theta_c5 = np.arctan2(
        tc.connector_link_lengths_lateral[tids.I_CONNECTOR_LINK_DFT_C5].item(),
        tc.connector_link_lengths_longitudinal[tids.I_CONNECTOR_LINK_DFT_C5].item(),
    )

    # initial direction
    if dft_state == "a" or dft_state == "d":
        gamma_c5 = np.arcsin(
            tc.pulley_radii[tids.I_RADIUS_DFT_5].item()
            / np.sqrt(
                tc.connector_link_lengths_lateral[tids.I_CONNECTOR_LINK_DFT_C5].item() ** 2
                + tc.connector_link_lengths_longitudinal[tids.I_CONNECTOR_LINK_DFT_C5].item() ** 2
            )
        )
    elif dft_state == "b":
        gamma_c5 = data["DFT_phi_4_B"]
    elif dft_state == "c":
        gamma_c5 = data["DFT_phi_4_C"]
    direction_angle = alpha_4 + theta_c5 - gamma_c5

    if dft_state == "a":
        p5_i = pc5 + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_DFT_C5].item(),
                    0.0,
                ]
            ),
        )

        q5 = data["qs"][tids.I_Q_DFT_5]

        p5_o = j5 + rotate_by(
            q5,
            p5_i - j5,
        )
        direction_angle += q5
        p6_i = p5_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_DFT_56].item(),
                    0.0,
                ]
            ),
        )
        q6 = data["qs"][tids.I_Q_DFT_6]
        p6_o = j6 + rotate_by(
            q6,
            p6_i - j6,
        )

        tendon_points = [pc5, p5_i, p5_o, p6_i, p6_o, p7]
        tendon_joints = [j5, j6]
        q_positives = [q5 >= 0, q6 >= 0]

    elif dft_state == "b":
        q6 = data["DFT_q6_B"]

        # First tangent point: connector C5 -> pulley j6
        p6_i = pc5 + rotate_by(
            direction_angle,
            np.array([data["DFT_l_c6"], 0.0]),
        )

        # Second tangent point: pulley j6 -> distal attachment p7
        # Compute it backwards from the known endpoint p7.
        out_dir = p7 - j6
        out_dir = out_dir / np.linalg.norm(out_dir)

        # Tangent radius is perpendicular to the outgoing tendon segment.
        # Try this sign first.
        p6_o = j6 + rotate_by(-np.pi / 2, out_dir) * tc.pulley_radii[tids.I_RADIUS_DFT_6].item()

        tendon_points = [pc5, p6_i, p6_o, p7]
        tendon_joints = [j6]
        q_positives = [q6 >= 0]
    # elif dft_state == "c":
    #     p7 = pc5 + rotate_by(direction_angle, np.array([np.sqrt(data["DFT_l_c7_squared"]), 0.0]))
    #     tendon_points = [pc5, p7]
    #     tendon_joints = []
    #     q_positives = []
    elif dft_state == "c":  # todo: check this
        tendon_points = [pc5, p7]
        tendon_joints = []
        q_positives = []
    # elif dft_state == "d":
    #     p5_i = pc5 + rotate_by(
    #         direction_angle,
    #         np.array(
    #             [
    #                 td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_DFT_C5].item(),
    #                 0.0,
    #             ]
    #         ),
    #     )
    #     q5 = data["DFT_q5_D"]
    #     p5_o = j5 + rotate_by(
    #         q5,
    #         p5_i - j5,
    #     )
    #     direction_angle += q5
    #     p7 = p5_o + rotate_by(
    #         direction_angle,
    #         np.array([data["DFT_l_57"], 0.0]),
    #     )
    #     tendon_points = [pc5, p5_i, p5_o, p7]
    #     tendon_joints = [j5]
    #     q_positives = [q5 >= 0]
    elif dft_state == "d":  # todo: check this
        p5_i = pc5 + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_DFT_C5].item(),
                    0.0,
                ]
            ),
        )
        q5 = data["DFT_q5_D"]
        p5_o = j5 + rotate_by(q5, p5_i - j5)

        tendon_points = [pc5, p5_i, p5_o, p7]
        tendon_joints = [j5]
        q_positives = [q5 >= 0]

    return tendon_points, tendon_joints, q_positives


def compute_edt1_points(alphas, joint_locations, data, r5):
    [_j2, _j3, j4, j5, j6, _p7] = joint_locations
    alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_8 = alphas
    edt1_state = get_edt1_state(data)
    # starting point is from j5 going offset length to j4 and orthogonally
    pc4 = j4 - rotate_by(
        alpha_3,
        np.array(
            [
                tc.connector_link_lengths_longitudinal[tids.I_CONNECTOR_LINK_EDT1_C4].item(),
                -tc.connector_link_lengths_lateral[tids.I_CONNECTOR_LINK_EDT1_C4].item(),
            ]
        ),
    )
    gamma_c4 = np.arctan2(
        tc.connector_link_lengths_lateral[tids.I_CONNECTOR_LINK_EDT1_C4].item(),
        tc.connector_link_lengths_longitudinal[tids.I_CONNECTOR_LINK_EDT1_C4].item(),
    )

    phi4a = data["EDT1_phi_4_a"]
    phi4b = data["EDT1_phi_4_b"] if edt1_state == "b" else np.pi / 2 - data["EDT1_phi_45_A"]

    phi4 = phi4a + phi4b
    direction_angle = alpha_3 + phi4 - gamma_c4
    if edt1_state == "a":
        p5_i = pc4 + rotate_by(direction_angle, np.array([data["EDT1_l_c5_A"], 0.0]))
        q5 = data["EDT1_q5_A"]
        p5_o = j5 + rotate_by(
            -q5,
            p5_i - j5,
        )
        direction_angle -= q5
        p6 = p5_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_EDT1_5C].item(),
                    0.0,
                ]
            ),
        )
        tendon_points = [pc4, p5_i, p5_o, p6]
        tendon_joints = [j5]
        q_positives = [q5 >= 0]
        # print(f"EDT1 q5: {q5}")
    else:
        p6c = pc4 + rotate_by(direction_angle, np.array([data["EDT1_l_cc"], 0.0]))
        tendon_points = [pc4, p6c]
        tendon_joints = []
        q_positives = []

    return tendon_points, tendon_joints, q_positives


def compute_edt2_points(alphas, joint_locations, data, r5, r6):
    [_j2, _j3, j4, j5, j6, p7] = joint_locations
    alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_8 = alphas
    edt2_state = get_edt2_state(data)
    # starting point is from j5 going offset length to j4 and orthogonally
    pc4 = j4 - rotate_by(
        alpha_3,
        np.array(
            [
                tc.connector_link_lengths_longitudinal[tids.I_CONNECTOR_LINK_EDT2_C4].item(),
                -tc.connector_link_lengths_lateral[tids.I_CONNECTOR_LINK_EDT2_C4].item(),
            ]
        ),
    )
    gamma_c4 = np.arctan2(
        tc.connector_link_lengths_lateral[tids.I_CONNECTOR_LINK_EDT2_C4].item(),
        tc.connector_link_lengths_longitudinal[tids.I_CONNECTOR_LINK_EDT2_C4].item(),
    )

    # initial direction
    if edt2_state == "a" or edt2_state == "d":
        phi4 = data["EDT2_phi_4_a"] + np.pi / 2 - data["EDT2_phi_45_A"]
    elif edt2_state == "b":
        # 3pi = (2pi - theta5) + (2pi - theta4) + phi6 + pi/2 + phi4
        phi4 = (
            -1.5 * np.pi
            - data["EDT2_phi_6_B"]
            + data["thetas"][tids.I_THETA_EDT2_5]
            + data["thetas"][tids.I_THETA_EDT2_4]
        )
    elif edt2_state == "c":
        phi4 = data["EDT2_phi_4_d"] + (np.pi - data["EDT2_thetatilde_6_a"] - data["EDT2_thetatilde_4"])
    direction_angle = alpha_3 + phi4 - gamma_c4

    if edt2_state == "a":
        p5_i = pc4 + rotate_by(direction_angle, np.array([data["EDT2_l_c5_A"], 0.0]))
        q5 = data["EDT2_q5_A"]
        p5_o = j5 + rotate_by(
            -q5,
            p5_i - j5,
        )
        direction_angle -= q5
        p6_i = p5_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_EDT2_56].item(),
                    0.0,
                ]
            ),
        )
        q6 = data["qhats"][tids.I_QHAT_EDT2_6]
        p6_o = j6 + rotate_by(
            -q6,
            p6_i - j6,
        )
        direction_angle -= q6
        p7 = p6_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_EDT2_6C].item(),
                    0.0,
                ]
            ),
        )

        tendon_points = [pc4, p5_i, p5_o, p6_i, p6_o, p7]
        tendon_joints = [j5, j6]
        q_positives = [q5 >= 0, q6 >= 0]
    elif edt2_state == "b":
        p6_i = pc4 + rotate_by(direction_angle, np.array([data["EDT2_l_c6_B"], 0.0]))
        q6 = data["EDT2_q6_B"]
        p6_o = j6 + rotate_by(
            -q6,
            p6_i - j6,
        )
        direction_angle -= q6
        p7 = p6_o + rotate_by(
            direction_angle,
            np.array(
                [
                    td.tendon_section_lengths[0, tids.I_TENDON_SECTION_LENGTH_EDT2_6C].item(),
                    0.0,
                ]
            ),
        )
        tendon_points = [pc4, p6_i, p6_o, p7]
        tendon_joints = [j6]
        q_positives = [q6 >= 0]
    elif edt2_state == "c":
        p7 = pc4 + rotate_by(direction_angle, np.array([data["EDT2_l_cc_C"], 0.0]))
        tendon_points = [pc4, p7]
        tendon_joints = []
        q_positives = []
    elif edt2_state == "d":
        p5_i = pc4 + rotate_by(direction_angle, np.array([data["EDT2_l_c5_A"], 0.0]))
        q5 = data["EDT2_q5_D"]
        p5_o = j5 + rotate_by(
            -q5,
            p5_i - j5,
        )
        direction_angle -= q5
        p7 = p5_o + rotate_by(
            direction_angle,
            np.array([data["EDT2_l_5c_D"], 0.0]),
        )
        tendon_points = [pc4, p5_i, p5_o, p7]
        tendon_joints = [j5]
        q_positives = [q5 >= 0]

    return tendon_points, tendon_joints, q_positives
