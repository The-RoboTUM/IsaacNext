"""Draws the dynamic tendon state with matplotlib animation.

Controls:
    Space: Play/Pause
    Left/Right Arrow: Step frame backward/forward (when paused)
    Home/End: Jump to first/last frame

Usage:
    python draw_tendon_action.py                    # Show animation
    python draw_tendon_action.py --save output.mp4  # Save to MP4
"""

import argparse
import copy
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from isaaclab.tendons.models.analytic.constants import (
    tids,
    TendonConstants,
    dummy_randomization,
)
from isaaclab.tendons.models.analytic.tendon_data import TendonData

tc = TendonConstants()
td = TendonData(1, dummy_randomization)

all_data = []
with open("outputs/gst_data_left.jsonl", "r") as f:
    for line in f:
        all_data.append(json.loads(line))


# draw the leg, starting at joint 2
alpha_2 = np.deg2rad(300)


def get_gst_state(data):
    state_a = data["GST_state_a"]
    state_b = data["GST_state_b"]
    state_c = data["GST_state_c"]
    state_d = data["GST_state_d"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    elif state_c:
        return "c"
    elif state_d:
        return "d"
    else:
        raise ValueError(f"GST: no state is true")


def get_dft_state(data):
    state_a = data["DFT_state_A"]
    state_b = data["DFT_state_B"]
    state_c = data["DFT_state_C"]
    state_d = data["DFT_state_D"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    elif state_c:
        return "c"
    elif state_d:
        return "d"
    else:
        raise ValueError(f"DFT: no state is true")


def get_edt1_state(data):
    state_a = data["EDT1_state_a"]
    state_b = data["EDT1_state_b"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    else:
        raise ValueError(f"EDT1: no state is true")


def get_edt2_state(data):
    state_a = data["EDT2_state_a"]
    state_b = data["EDT2_state_b"]
    state_c = data["EDT2_state_c"]
    state_d = data["EDT2_state_d"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    elif state_c:
        return "c"
    elif state_d:
        return "d"
    else:
        raise ValueError(f"EDT2: no state is true")


def compute_alphas(alpha_2, thetas):
    theta_3, theta_4, theta_5, theta_6, theta_8 = (
        thetas[tids.I_THETA_GST_3],
        thetas[tids.I_THETA_GST_4],
        thetas[tids.I_THETA_GST_5],
        thetas[tids.I_THETA_ALL_6],
        thetas[tids.I_THETA_KFT_8],
    )
    alpha_3 = np.pi + alpha_2 - theta_3
    alpha_4 = alpha_3 + theta_4 - np.pi
    alpha_5 = alpha_4 + theta_5 - np.pi
    alpha_6 = alpha_5 + theta_6 - np.pi
    alpha_8 = alpha_2 + theta_8 - 2 * np.pi
    return [alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_8]


def rotate_by(angle, vector):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s], [s, c]]) @ vector


def compute_joint_locations(alphas):
    alpha_2, alpha_3, alpha_4, alpha_5, alpha_6, alpha_8 = alphas
    j2 = np.zeros(2)
    j3 = j2 + rotate_by(
        alpha_2,
        np.array([td.link_lengths[0, tids.I_CONNECTOR_LINK_GST_23].item(), 0.0]),
    )
    j4 = j3 + rotate_by(alpha_3, np.array([td.link_lengths[0, tids.I_LINK_34].item(), 0.0]))
    j5 = j4 + rotate_by(alpha_4, np.array([td.link_lengths[0, tids.I_LINK_4prime5].item(), 0.0]))
    j6 = j5 + rotate_by(alpha_5, np.array([td.link_lengths[0, tids.I_LINK_56].item(), 0.0]))
    p7 = j6 + rotate_by(alpha_6, np.array([td.link_lengths[0, tids.I_LINK_67].item(), 0.0]))

    return [j2, j3, j4, j5, j6, p7]


# takes alpha_2, beta_2, assumes starting point at r(49.5°) @ [0.059407 0]
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
    assert (
        abs(radius_4 - td.pulley_radii[0, tids.I_RADIUS_GST_4].item()) < 0.001
    ), f"Expected radius at 4 {td.pulley_radii[0, tids.I_RADIUS_GST_4].item()}, got {radius_4}"
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
    upper_tendon_q_positives = [q3 >= 0, q4 >= 0, q4 >= 0]

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
        p6_i = pc5 + rotate_by(
            direction_angle,
            np.array(
                [
                    data["DFT_l_c6"],
                    0.0,
                ]
            ),
        )
        q6 = data["DFT_q6_B"]
        p6_o = j6 + rotate_by(
            q6,
            p6_i - j6,
        )

        tendon_points = [pc5, p6_i, p6_o, p7]
        tendon_joints = [j6]
        q_positives = [q6 >= 0]
    elif dft_state == "c":
        p7 = pc5 + rotate_by(direction_angle, np.array([np.sqrt(data["DFT_l_c7_squared"]), 0.0]))
        tendon_points = [pc5, p7]
        tendon_joints = []
        q_positives = []
    elif dft_state == "d":
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
        p5_o = j5 + rotate_by(
            q5,
            p5_i - j5,
        )
        direction_angle += q5
        p7 = p5_o + rotate_by(
            direction_angle,
            np.array([data["DFT_l_57"], 0.0]),
        )
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


# for state A: none, state B: x4'6 with h5B, state C: x4'6, x4'7, h5C, h6C, state D: x57, h6D; draw using phis
def validate_xs(joint_locations, x4prime6, x4prime7, x57):
    [_j2, _j3, j4, j5, j6, p7] = joint_locations
    assert abs(np.linalg.norm(j6 - j4) - x4prime6) < 0.001, f"expected x4'6={np.linalg.norm(j6 - j4)}, got {x4prime6}"
    assert abs(np.linalg.norm(p7 - j4) - x4prime7) < 0.001, f"expected x4'7={np.linalg.norm(p7 - j4)}, got {x4prime7}"
    assert abs(np.linalg.norm(p7 - j5) - x57) < 0.001, f"expected x57={np.linalg.norm(p7 - j5)}, got {x57}"


def validate_ls(joint_locations, l4prime6, l4prime7, l57):
    [_j2, _j3, j4, j5, j6, p7] = joint_locations
    x46_inf = np.linalg.norm(j6 - j4)
    l46_inf = np.sqrt(
        x46_inf**2 - ((tc.pulley_radii[tids.I_RADIUS_GST_4prime] - tc.pulley_radii[tids.I_RADIUS_GST_6]) ** 2).item()
    )
    assert abs(l46_inf - l4prime6) < 0.001, f"Expected l4'6={l46_inf}, got {l4prime6}"

    x47_inf = np.linalg.norm(p7 - j4)
    l47_inf = np.sqrt(x47_inf**2 - (tc.pulley_radii[tids.I_RADIUS_GST_4prime] ** 2).item())
    assert abs(l47_inf - l4prime7) < 0.001, f"Expected l4'7={l47_inf}, got {l4prime7}"

    x57_inf = np.linalg.norm(p7 - j5)
    l57_inf = np.sqrt(x57_inf**2 - (tc.pulley_radii[tids.I_RADIUS_GST_5] ** 2).item())
    assert abs(l57_inf - l57) < 0.001, f"Expected l57={l57_inf}, got {l57}"


def arc_from_3_points(c, p1, p2, ccw=True, q_positive=True, tol=1e-4):
    d1 = p1 - c
    d2 = p2 - c
    assert np.isclose(np.linalg.norm(d1), np.linalg.norm(d2), atol=tol), "start and end points should have same radius"
    r = np.linalg.norm(d1)
    # assert np.isclose(d1, r, atol=tol), f"Start point not on circle: {d1} != {r}"
    # assert np.isclose(d2, r, atol=tol), f"End point not on circle: {d2} != {r}"
    theta_1 = np.arctan2(d1[1], d1[0])
    theta_2 = np.arctan2(d2[1], d2[0])
    if theta_2 < theta_1 and ccw and q_positive:
        theta_2 += 2 * np.pi
    if theta_1 < theta_2 and ccw and not q_positive:
        theta_1 += 2 * np.pi
    # if theta_1 < theta_2 and not ccw and q_positive:
    #     theta_1 += 2 * np.pi
    if theta_2 < theta_1 and not ccw and not q_positive:
        theta_2 += 2 * np.pi

    thetas = np.linspace(theta_1, theta_2, 20)

    xs = r * np.cos(thetas) + c[0]
    ys = r * np.sin(thetas) + c[1]

    return xs, ys


def compute_x_end_point(link_start, link_end, phi, x, goal_point):
    link_dir = link_end - link_start
    rotated = rotate_by(phi, link_dir)
    scaled = rotated / np.linalg.norm(rotated) * x
    assert np.allclose(
        scaled + link_start, goal_point, rtol=1e-3, atol=1e-3
    ), f"Expected {goal_point}, got {scaled + link_start}"
    return scaled + link_start


def compute_h_end_point(tendon_start, tendon_end, joint, height):
    # Direction vector of the tendon line
    d = tendon_end - tendon_start
    # Vector from tendon_start to joint
    v = joint - tendon_start
    # Projection parameter
    t = np.dot(v, d) / np.dot(d, d)
    # Projected point on the line
    projected = tendon_start + t * d
    # Orthogonal distance from joint to line
    dist = np.linalg.norm(joint - projected)
    # assert abs(dist - height) < 0.001, f"Expected height {height}, got {dist}"
    return projected


class KinematicChainAnimator:
    def __init__(self, all_data, alpha_2=np.deg2rad(300)):
        all_thetas = [d["thetas"] for d in all_data]
        self.all_thetas = np.array(all_thetas)
        self.alpha_2 = alpha_2
        self.num_frames = len(all_thetas)
        self.current_frame = 0
        self.is_playing = True

        # Set up the figure
        self.fig, self.ax = plt.subplots(2, 2, figsize=(11, 11))
        self.n_ax = 4
        for ax in self.ax.flat:
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

        # Initialize plot elements
        self.skeleton_lines = []
        self.joints_scatters = []
        self.end_effectors = []
        for ax in self.ax.flat:
            (skeleton_line,) = ax.plot([], [], "b-", linewidth=2, label="Links")
            self.skeleton_lines.append(skeleton_line)
            joints_scatter = ax.scatter([], [], s=5, c="red", zorder=5, label="Joints")
            self.joints_scatters.append(joints_scatter)
            (end_effector,) = ax.plot([], [], "go", markersize=4, label="End effector")
            self.end_effectors.append(end_effector)

        helper_line_alpha = 0.7

        current_data = all_data[0]
        gst_state = get_gst_state(current_data)
        thetas = self.all_thetas[0]
        alphas = compute_alphas(self.alpha_2, thetas)
        joints = compute_joint_locations(alphas)
        j2, j3, j4, j5, j6, p7 = joints

        # Pulley circles (created once and reused)
        self.gst_pulley_circles = {}
        self.kft_dft_pulley_circles = {}
        self.edt1_pulley_circles = {}
        self.edt2_pulley_circles = {}

        gst_pulley_configs = [
            (j3, tc.pulley_radii[tids.I_RADIUS_GST_3].item(), "orange", "3"),
            (j4, tc.pulley_radii[tids.I_RADIUS_GST_4].item(), "purple", "4"),
            (j4, tc.pulley_radii[tids.I_RADIUS_GST_4prime].item(), "magenta", "4prime"),
            (j5, tc.pulley_radii[tids.I_RADIUS_GST_5].item(), "cyan", "5"),
            (j6, tc.pulley_radii[tids.I_RADIUS_GST_6].item(), "navy", "6"),
        ]
        kft_pulley_configs = [
            (j2, tc.pulley_radii[tids.I_RADIUS_KFT_8].item(), "brown", "8"),
        ]
        dft_pulley_configs = [
            (j5, tc.pulley_radii[tids.I_RADIUS_DFT_5].item(), "cyan", "5"),
            (j6, tc.pulley_radii[tids.I_RADIUS_DFT_6].item(), "navy", "6"),
        ]
        edt1_pulley_configs = [
            (j5, tc.pulley_radii[tids.I_RADIUS_EDT1_5].item(), "cyan", "5"),
        ]
        edt2_pulley_configs = [
            (j5, tc.pulley_radii[tids.I_RADIUS_EDT2_5].item(), "cyan", "5"),
            (j6, tc.pulley_radii[tids.I_RADIUS_EDT2_6].item(), "navy", "6"),
        ]
        for center, radius, color, label in gst_pulley_configs:
            circle = Circle(center, radius, fill=False, edgecolor=color, linewidth=1.5, alpha=0.7)
            self.ax[0, 0].add_patch(circle)
            self.gst_pulley_circles[label] = circle
        for center, radius, color, label in kft_pulley_configs + dft_pulley_configs:
            circle = Circle(center, radius, fill=False, edgecolor=color, linewidth=1.5, alpha=0.7)
            self.ax[0, 1].add_patch(circle)
            self.kft_dft_pulley_circles[label] = circle
        for center, radius, color, label in edt1_pulley_configs:
            circle = Circle(center, radius, fill=False, edgecolor=color, linewidth=1.5, alpha=0.7)
            self.ax[1, 0].add_patch(circle)
            self.edt1_pulley_circles[label] = circle
        for center, radius, color, label in edt2_pulley_configs:
            circle = Circle(center, radius, fill=False, edgecolor=color, linewidth=1.5, alpha=0.7)
            self.ax[1, 1].add_patch(circle)
            self.edt2_pulley_circles[label] = circle

        # GST-specific lines
        (self.gst_upper_tendon_line,) = self.ax[0, 0].plot([], [], "r-", linewidth=2, label="Upper Tendon")
        (self.gst_lower_tendon_line,) = self.ax[0, 0].plot([], [], "y-", linewidth=2, label="Lower Tendon")

        (self.gst_x_4prime6_line,) = self.ax[0, 0].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x4'6",
            alpha=helper_line_alpha,
        )
        (self.gst_x_4prime7_line,) = self.ax[0, 0].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x4'7",
            alpha=helper_line_alpha,
        )
        (self.gst_x_57_line,) = self.ax[0, 0].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x57",
            alpha=helper_line_alpha,
        )
        (self.gst_h_5_line,) = self.ax[0, 0].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h5",
            alpha=helper_line_alpha,
        )
        (self.gst_h_6_line,) = self.ax[0, 0].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h6",
            alpha=helper_line_alpha,
        )

        # KFT/DFT-specific lines
        (self.kft_tendon_line,) = self.ax[0, 1].plot([], [], "r-", linewidth=2, label="Upper Tendon")
        (self.dft_tendon_line,) = self.ax[0, 1].plot([], [], "-", linewidth=2, label="Lower Tendon", color="orange")
        (self.dft_x_c6_line,) = self.ax[0, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x4'6",
            alpha=helper_line_alpha,
        )
        (self.dft_x_57_line,) = self.ax[0, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x57",
            alpha=helper_line_alpha,
        )
        (self.dft_h_5_line,) = self.ax[0, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h5",
            alpha=helper_line_alpha,
        )
        (self.dft_h_6_line,) = self.ax[0, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h6",
            alpha=helper_line_alpha,
        )

        # EDT1 lines
        (self.edt1_tendon_line,) = self.ax[1, 0].plot([], [], "r-", linewidth=2, label="EDT1 Tendon")
        # xc5,  h5
        (self.edt1_h_5_line,) = self.ax[1, 0].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h5",
            alpha=helper_line_alpha,
        )
        (self.edt1_x_c5_line,) = self.ax[1, 0].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="xc5",
            alpha=helper_line_alpha,
        )

        # EDT2 lines
        (self.edt2_tendon_line,) = self.ax[1, 1].plot([], [], "y-", linewidth=2, label="EDT2 Tendon")
        # h5, h6, xc5, xc6 x5c
        (self.edt2_h_5_line,) = self.ax[1, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h5",
            alpha=helper_line_alpha,
        )
        (self.edt2_h_6_line,) = self.ax[1, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h6",
            alpha=helper_line_alpha,
        )
        (self.edt2_x_c5_line,) = self.ax[1, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="xc5",
            alpha=helper_line_alpha,
        )
        (self.edt2_x_c6_line,) = self.ax[1, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="xc6",
            alpha=helper_line_alpha,
        )
        (self.edt2_x_46_line,) = self.ax[1, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x46",
            alpha=helper_line_alpha,
        )
        (self.edt2_x_5c_line,) = self.ax[1, 1].plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x5c",
            alpha=helper_line_alpha,
        )

        # Title and info text
        self.gst_title = self.ax[0, 0].set_title("")
        self.gst_info_text = self.ax[0, 0].text(
            0.02,
            0.98,
            "",
            transform=self.ax[0, 0].transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        # Delta l text in top center with larger font
        self.gst_delta_l_text = self.ax[0, 0].text(
            0.5,
            0.98,
            "",
            transform=self.ax[0, 0].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontfamily="monospace",
            fontsize=16,
        )
        self.kft_delta_l_text = self.ax[0, 1].text(
            0.5,
            0.98,
            "",
            transform=self.ax[0, 1].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontfamily="monospace",
            fontsize=16,
        )
        self.dft_delta_l_text = self.ax[0, 1].text(
            0.5,
            0.90,
            "",
            transform=self.ax[0, 1].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontfamily="monospace",
            fontsize=16,
        )
        self.edt1_delta_l_text = self.ax[1, 0].text(
            0.5,
            0.98,
            "",
            transform=self.ax[1, 0].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontfamily="monospace",
            fontsize=16,
        )

        # EDT2 state text (lower-right)
        self.edt2_state_text = self.ax[1, 1].text(
            0.98,
            0.02,
            "",
            transform=self.ax[1, 1].transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        # EDT2 delta l text in top center with larger font
        self.edt2_delta_l_text = self.ax[1, 1].text(
            0.5,
            0.98,
            "",
            transform=self.ax[1, 1].transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontfamily="monospace",
            fontsize=16,
        )

        # Compute axis limits from all frames
        self._compute_axis_limits()

        # Set up animation
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self._frame_generator,
            interval=50,
            blit=True,
            save_count=self.num_frames,
        )

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        for ax in self.ax.flat:
            ax.legend(loc="upper right")
        self.fig.tight_layout()

    def _compute_axis_limits(self):
        """Pre-compute axis limits based on all frames."""
        all_x, all_y = [], []
        for thetas in self.all_thetas:
            alphas = compute_alphas(self.alpha_2, thetas)
            joints = compute_joint_locations(alphas)
            for j in joints:
                all_x.append(j[0])
                all_y.append(j[1])

        margin = 0.15
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        for ax in self.ax.flat:
            ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
            ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)

    def _frame_generator(self):
        """Generator that yields frame indices, respecting pause state."""
        while True:
            if self.is_playing:
                self.current_frame = (self.current_frame + 1) % self.num_frames
            yield self.current_frame

    def tendon_path(self, tendon_points: list, tendon_joints: list, upper_tendon=True):
        arc = not upper_tendon
        last_point = tendon_points.pop(0)
        xs = [last_point[0]]
        ys = [last_point[1]]
        while len(tendon_points) > 0:
            current_point = tendon_points.pop(0)
            if not arc:
                xs.append(current_point[0])
                ys.append(current_point[1])
                arc = True
            else:
                current_joint = tendon_joints.pop(0)

                new_xs, new_ys = arc_from_3_points(
                    current_joint,
                    last_point,
                    current_point,
                    ccw=not (upper_tendon and len(tendon_joints) == 1),
                )
                xs.extend(new_xs)
                ys.extend(new_ys)
                arc = False
            last_point = current_point

        return xs, ys

    def tendon_path_general(
        self,
        tendon_points: list,
        tendon_joints: list,
        tendon_q_positives: list,
        joint_ccws: list,
        start_with_arc=False,
    ):
        arc = start_with_arc
        last_point = tendon_points.pop(0)
        xs = [last_point[0]]
        ys = [last_point[1]]
        while len(tendon_points) > 0:
            current_point = tendon_points.pop(0)
            if not arc:
                xs.append(current_point[0])
                ys.append(current_point[1])
                arc = True
            else:
                current_joint = tendon_joints.pop(0)
                current_q_positive = tendon_q_positives.pop(0)
                current_joint_ccw = joint_ccws.pop(0)

                new_xs, new_ys = arc_from_3_points(
                    current_joint,
                    last_point,
                    current_point,
                    ccw=current_joint_ccw,
                    q_positive=current_q_positive,
                )
                xs.extend(new_xs)
                ys.extend(new_ys)
                arc = False
            last_point = current_point

        return xs, ys

    def _update(self, frame_idx):
        """Update function for animation."""
        if frame_idx % 100 == 0:
            print(f"frame {frame_idx}")
        current_data = all_data[frame_idx]
        gst_state = get_gst_state(current_data)
        thetas = self.all_thetas[frame_idx]
        alphas = compute_alphas(self.alpha_2, thetas)
        joints = compute_joint_locations(alphas)

        j2, j3, j4, j5, j6, p7 = joints

        # Update skeleton line (connecting all joints)
        x_coords = [j2[0], j3[0], j4[0], j5[0], j6[0], p7[0]]
        y_coords = [j2[1], j3[1], j4[1], j5[1], j6[1], p7[1]]
        for skeleton_line in self.skeleton_lines:
            skeleton_line.set_data(x_coords, y_coords)
        joint_x = [j[0] for j in joints[:-1]]  # Exclude end effector
        joint_y = [j[1] for j in joints[:-1]]
        for joints_scatter in self.joints_scatters:
            # Update joint positions
            joints_scatter.set_offsets(np.c_[joint_x, joint_y])

        # Update end effector
        for end_effector in self.end_effectors:
            end_effector.set_data([p7[0]], [p7[1]])

        # Update pulley circle positions
        self.gst_pulley_circles["3"].center = j3
        self.gst_pulley_circles["4"].center = j4
        self.gst_pulley_circles["4prime"].center = j4
        self.gst_pulley_circles["5"].center = j5
        self.gst_pulley_circles["6"].center = j6
        self.kft_dft_pulley_circles["8"].center = j2
        self.kft_dft_pulley_circles["5"].center = j5
        self.kft_dft_pulley_circles["6"].center = j6
        self.edt1_pulley_circles["5"].center = j5
        self.edt2_pulley_circles["5"].center = j5
        self.edt2_pulley_circles["6"].center = j6

        # Draw GST tendons
        (
            upper_tendon_points,
            upper_tendon_joints,
            upper_q_positives,
            lower_tendon_points,
            lower_tendon_joints,
            lower_q_positives,
        ) = compute_gst_attachment_points(alpha_2, joints, current_data)
        upper_gst_xs, upper_gst_ys = self.tendon_path(upper_tendon_points, upper_tendon_joints, upper_tendon=True)
        lower_gst_xs, lower_gst_ys = self.tendon_path(
            copy.copy(lower_tendon_points), lower_tendon_joints, upper_tendon=False
        )
        self.gst_upper_tendon_line.set_data(upper_gst_xs, upper_gst_ys)
        self.gst_lower_tendon_line.set_data(lower_gst_xs, lower_gst_ys)

        # validate gst data
        validate_xs(
            joints,
            current_data["GST_x_4prime6"],
            current_data["GST_x_4prime7"],
            current_data["GST_x_57"],
        )
        validate_ls(
            joints,
            current_data["GST_l_4prime6"],
            current_data["GST_l_4prime7"],
            current_data["GST_l_57"],
        )

        # Draw GST helper lines (x's and h's)
        self.gst_x_4prime6_line.set_data([], [])
        self.gst_x_4prime7_line.set_data([], [])
        self.gst_x_57_line.set_data([], [])
        self.gst_h_5_line.set_data([], [])
        self.gst_h_6_line.set_data([], [])
        if gst_state == "b":
            x_4prime6_end_point = compute_x_end_point(
                j4,
                j5,
                -current_data["GST_phi_4prime_a"],
                current_data["GST_x_4prime6"],
                goal_point=j6,
            )  # rotate 4'5 by -phi_4'a and scale
            self.gst_x_4prime6_line.set_data([j4[0], x_4prime6_end_point[0]], [j4[1], x_4prime6_end_point[1]])
            h5_end_point = compute_h_end_point(
                lower_tendon_points[1],
                lower_tendon_points[2],
                j5,
                current_data["GST_h5_B"],
            )  # project j5 onto according line segment, assert distance is equal to h5B
            self.gst_h_5_line.set_data([j5[0], h5_end_point[0]], [j5[1], h5_end_point[1]])
        elif gst_state == "c":
            x_4prime6_end_point = compute_x_end_point(
                j4,
                j5,
                -current_data["GST_phi_4prime_a"],
                current_data["GST_x_4prime6"],
                goal_point=j6,
            )  # rotate 4'5 by -phi_4'a and scale
            self.gst_x_4prime6_line.set_data([j4[0], x_4prime6_end_point[0]], [j4[1], x_4prime6_end_point[1]])
            h5_end_point = compute_h_end_point(
                lower_tendon_points[1],
                lower_tendon_points[2],
                j5,
                current_data["GST_h5_C"],
            )  # project j5 onto according line segment, assert distance is equal to h5C
            self.gst_h_5_line.set_data([j5[0], h5_end_point[0]], [j5[1], h5_end_point[1]])

            x_4prime7_end_point = compute_x_end_point(
                j4,
                j5,
                -current_data["GST_phi_4prime_a"] - current_data["GST_phi_4prime_d"],
                current_data["GST_x_4prime7"],
                goal_point=p7,
            )  # rotate 4'5 by -phi_4'a-phi_4'd and scale
            self.gst_x_4prime7_line.set_data([j4[0], x_4prime7_end_point[0]], [j4[1], x_4prime7_end_point[1]])
            h6_end_point = compute_h_end_point(
                lower_tendon_points[1],
                lower_tendon_points[2],
                j6,
                current_data["GST_h6_C"],
            )  # project j6 onto according line segment, assert distance is equal to h6C
            self.gst_h_6_line.set_data([j6[0], h6_end_point[0]], [j6[1], h6_end_point[1]])

        elif gst_state == "d":
            x_57_end_point = compute_x_end_point(
                j5,
                j6,
                -current_data["GST_phi_5_a"],
                current_data["GST_x_57"],
                goal_point=p7,
            )  # rotate 56 by -phi_5a and scale
            self.gst_x_57_line.set_data([j5[0], x_57_end_point[0]], [j5[1], x_57_end_point[1]])
            h6_end_point = compute_h_end_point(
                lower_tendon_points[-2],
                lower_tendon_points[-1],
                j6,
                current_data["GST_h6_D"],
            )  # project j6 onto according line segment, assert distance is equal to h6B
            self.gst_h_6_line.set_data([j6[0], h6_end_point[0]], [j6[1], h6_end_point[1]])

        # Draw DFT
        dft_points, dft_joints, dft_q_positives = compute_dft_points(
            alphas,
            joints,
            current_data,
            tc.pulley_radii[tids.I_RADIUS_DFT_5].item(),
            tc.pulley_radii[tids.I_RADIUS_DFT_6].item(),
        )

        # Draw helper lines for DFT
        dft_state = get_dft_state(current_data)
        if dft_state == "a":
            self.dft_h_5_line.set_data([], [])
            self.dft_h_6_line.set_data([], [])
            self.dft_x_c6_line.set_data([], [])
            self.dft_x_57_line.set_data([], [])
        elif dft_state == "b":
            dft_h_5_end_point = compute_h_end_point(
                dft_points[0],
                dft_points[1],
                j5,
                current_data["DFT_h5_B"],
            )
            self.dft_h_5_line.set_data([j5[0], dft_h_5_end_point[0]], [j5[1], dft_h_5_end_point[1]])
            self.dft_h_6_line.set_data([], [])
            self.dft_x_c6_line.set_data([dft_points[0][0], j6[0]], [dft_points[0][1], j6[1]])
            self.dft_x_57_line.set_data([], [])
        elif dft_state == "d":
            dft_h_6_end_point = compute_h_end_point(
                dft_points[-2],
                dft_points[-1],
                j6,
                current_data["DFT_h6_D"],
            )
            self.dft_h_5_line.set_data([], [])
            self.dft_h_6_line.set_data([j6[0], dft_h_6_end_point[0]], [j6[1], dft_h_6_end_point[1]])
            self.dft_x_c6_line.set_data([], [])
            self.dft_x_57_line.set_data([j5[0], p7[0]], [j5[1], p7[1]])
        elif dft_state == "c":
            dft_h_5_end_point = compute_h_end_point(
                dft_points[0],
                dft_points[1],
                j5,
                current_data["DFT_h5_C"],
            )
            dft_h_6_end_point = compute_h_end_point(
                dft_points[0],
                dft_points[1],
                j6,
                current_data["DFT_h6_C"],
            )
            self.dft_h_5_line.set_data([j5[0], dft_h_5_end_point[0]], [j5[1], dft_h_5_end_point[1]])
            self.dft_h_6_line.set_data([j6[0], dft_h_6_end_point[0]], [j6[1], dft_h_6_end_point[1]])
            self.dft_x_c6_line.set_data([dft_points[0][0], j6[0]], [dft_points[0][1], j6[1]])
            self.dft_x_57_line.set_data([], [])

        # Draw DFT tendon
        dft_xs, dft_ys = self.tendon_path_general(
            dft_points,
            dft_joints,
            dft_q_positives,
            [True, True],
            start_with_arc=False,
        )
        self.dft_tendon_line.set_data(dft_xs, dft_ys)

        # Draw KFT
        kft_points, kft_joints, kft_q_positives = compute_kft_points(
            alphas[5], joints, current_data, tc.pulley_radii[tids.I_RADIUS_KFT_8].item()
        )
        kft_xs, kft_ys = self.tendon_path_general(kft_points, kft_joints, kft_q_positives, [False], start_with_arc=True)
        self.kft_tendon_line.set_data(kft_xs, kft_ys)

        # Compute EDT1
        edt1_points, edt1_joints, edt1_q_positives = compute_edt1_points(
            alphas,
            joints,
            current_data,
            tc.pulley_radii[tids.I_RADIUS_EDT1_5].item(),
        )

        # Draw helper lines for EDT1
        self.edt1_x_c5_line.set_data([edt1_points[0][0], j5[0]], [edt1_points[0][1], j5[1]])
        if get_edt1_state(current_data) == "b":
            edt1_h_5_end_point = compute_h_end_point(
                edt1_points[0],
                edt1_points[1],
                j5,
                current_data["EDT1_h5_B"],
            )
            self.edt1_h_5_line.set_data([j5[0], edt1_h_5_end_point[0]], [j5[1], edt1_h_5_end_point[1]])

        else:
            self.edt1_h_5_line.set_data([], [])

        # Draw EDT1 tendon
        edt1_xs, edt1_ys = self.tendon_path_general(
            edt1_points, edt1_joints, edt1_q_positives, [False], start_with_arc=False
        )
        self.edt1_tendon_line.set_data(edt1_xs, edt1_ys)

        # Compute EDT2
        edt2_points, edt2_joints, edt2_q_positives = compute_edt2_points(
            alphas,
            joints,
            current_data,
            tc.pulley_radii[tids.I_RADIUS_EDT2_5].item(),
            tc.pulley_radii[tids.I_RADIUS_EDT2_6].item(),
        )

        # Draw helper lines for EDT2
        self.edt2_x_c5_line.set_data([edt2_points[0][0], j5[0]], [edt2_points[0][1], j5[1]])
        edt2_state = get_edt2_state(current_data)
        if edt2_state == "a":
            self.edt2_h_5_line.set_data([], [])
            self.edt2_h_6_line.set_data([], [])
            self.edt2_x_c6_line.set_data([], [])
            self.edt2_x_46_line.set_data([], [])
            self.edt2_x_5c_line.set_data([], [])
        elif edt2_state == "b":
            edt2_h_5_end_point = compute_h_end_point(
                edt2_points[0],
                edt2_points[1],
                j5,
                current_data["EDT2_h5_B"],
            )
            self.edt2_h_5_line.set_data([j5[0], edt2_h_5_end_point[0]], [j5[1], edt2_h_5_end_point[1]])
            self.edt2_h_6_line.set_data([], [])
            self.edt2_x_c6_line.set_data([edt2_points[0][0], j6[0]], [edt2_points[0][1], j6[1]])
            self.edt2_x_46_line.set_data([], [])
            self.edt2_x_5c_line.set_data([], [])
        elif edt2_state == "d":
            edt2_h_6_end_point = compute_h_end_point(
                edt2_points[-2],
                edt2_points[-1],
                j6,
                current_data["EDT2_h6_D"],
            )
            self.edt2_h_5_line.set_data([], [])
            self.edt2_h_6_line.set_data([j6[0], edt2_h_6_end_point[0]], [j6[1], edt2_h_6_end_point[1]])
            self.edt2_x_c6_line.set_data([], [])
            self.edt2_x_46_line.set_data([], [])
            self.edt2_x_5c_line.set_data([j5[0], p7[0]], [j5[1], p7[1]])
        elif edt2_state == "c":
            edt2_h_5_end_point = compute_h_end_point(
                edt2_points[0],
                edt2_points[1],
                j5,
                current_data["EDT2_h5_C"],
            )
            edt2_h_6_end_point = compute_h_end_point(
                edt2_points[0],
                edt2_points[1],
                j6,
                current_data["EDT2_h6_C"],
            )
            self.edt2_h_5_line.set_data([j5[0], edt2_h_5_end_point[0]], [j5[1], edt2_h_5_end_point[1]])
            self.edt2_h_6_line.set_data([j6[0], edt2_h_6_end_point[0]], [j6[1], edt2_h_6_end_point[1]])
            self.edt2_x_c6_line.set_data([edt2_points[0][0], j6[0]], [edt2_points[0][1], j6[1]])
            self.edt2_x_46_line.set_data([j4[0], j6[0]], [j4[1], j6[1]])
            self.edt2_x_5c_line.set_data([], [])

        # Draw EDT2 tendon
        edt2_xs, edt2_ys = self.tendon_path_general(
            edt2_points,
            edt2_joints,
            edt2_q_positives,
            [False, False],
            start_with_arc=False,
        )
        self.edt2_tendon_line.set_data(edt2_xs, edt2_ys)

        # Update title and info
        status = "▶ Playing" if self.is_playing else "⏸ Paused"
        gst_state = get_gst_state(all_data[frame_idx])
        self.gst_title.set_text(f"Kinematic Chain Animation - Frame {frame_idx + 1}/{self.num_frames}")
        self.gst_info_text.set_text(
            f"{status}\n"
            f"GST State: {gst_state}\n"
            f"θ₃: {np.rad2deg(thetas[0]):.1f}°\n"
            f"θ₄: {np.rad2deg(thetas[1]):.1f}°\n"
            f"θ₅: {np.rad2deg(thetas[2]):.1f}°\n"
            f"θ₆: {np.rad2deg(thetas[3]):.1f}°"
        )
        # Update delta l text with conditional coloring
        delta_l = all_data[frame_idx]["GST_delta_L_s"]
        delta_l_color = "grey" if delta_l > 0 else "green"
        self.gst_delta_l_text.set_text(f"$\\Delta L={delta_l * 1000:02.3f}$ mm")
        self.gst_delta_l_text.set_color(delta_l_color)

        kft_delta_l = all_data[frame_idx]["KFT_delta_L_s"]
        kft_delta_l_color = "grey" if kft_delta_l > 0 else "green"
        self.kft_delta_l_text.set_text(f"KFT $\\Delta L={kft_delta_l * 1000:02.3f}$ mm")
        self.kft_delta_l_text.set_color(kft_delta_l_color)

        dft_delta_l = all_data[frame_idx]["DFT_delta_L_s"]
        dft_delta_l_color = "grey" if dft_delta_l > 0 else "green"
        self.dft_delta_l_text.set_text(f"DFT $\\Delta L={dft_delta_l * 1000:02.3f}$ mm")
        self.dft_delta_l_text.set_color(dft_delta_l_color)

        edt1_delta_l = all_data[frame_idx]["EDT1_delta_L_s"]
        edt1_delta_l_color = "grey" if edt1_delta_l > 0 else "green"
        self.edt1_delta_l_text.set_text(f"EDT1 $\\Delta L={edt1_delta_l * 1000:02.3f}$ mm")
        self.edt1_delta_l_text.set_color(edt1_delta_l_color)

        edt2_state = get_edt2_state(all_data[frame_idx])
        self.edt2_state_text.set_text(f"EDT2 State: {edt2_state}")

        edt2_delta_l = all_data[frame_idx]["EDT2_delta_L_s"]
        edt2_delta_l_color = "grey" if edt2_delta_l > 0 else "green"
        self.edt2_delta_l_text.set_text(f"EDT2 $\\Delta L={edt2_delta_l * 1000:02.3f}$ mm")
        self.edt2_delta_l_text.set_color(edt2_delta_l_color)

        # Return all artists for blitting
        artists = (
            self.skeleton_lines
            + self.joints_scatters
            + self.end_effectors
            + [self.gst_upper_tendon_line, self.gst_lower_tendon_line]
            + [self.gst_x_4prime6_line, self.gst_x_4prime7_line, self.gst_x_57_line]
            + [self.gst_h_5_line, self.gst_h_6_line]
            + [self.kft_tendon_line, self.dft_tendon_line]
            + list(self.gst_pulley_circles.values())
            + list(self.kft_dft_pulley_circles.values())
            + list(self.edt1_pulley_circles.values())
            + list(self.edt2_pulley_circles.values())
            + [self.gst_title, self.gst_info_text, self.gst_delta_l_text]
            + [self.kft_delta_l_text, self.dft_delta_l_text, self.edt1_delta_l_text]
            + [self.edt2_state_text, self.edt2_delta_l_text]
            + [self.edt1_tendon_line, self.edt1_h_5_line, self.edt1_x_c5_line]
            + [
                self.edt2_tendon_line,
                self.edt2_h_5_line,
                self.edt2_h_6_line,
                self.edt2_x_c5_line,
                self.edt2_x_c6_line,
                self.edt2_x_46_line,
                self.edt2_x_5c_line,
            ]
            + [
                self.dft_x_57_line,
                self.dft_x_c6_line,
                self.dft_h_5_line,
                self.dft_h_6_line,
            ]
        )
        return artists

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == " ":
            self.is_playing = not self.is_playing
        elif event.key == "right" and not self.is_playing:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "left" and not self.is_playing:
            self.current_frame = (self.current_frame - 1) % self.num_frames
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "home":
            self.current_frame = 0
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "end":
            self.current_frame = self.num_frames - 1
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()

    def show(self):
        """Display the animation."""
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate tendon kinematics")
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="Save animation to MP4 file instead of displaying",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for saved video (default: 30)",
    )
    args = parser.parse_args()

    animator = KinematicChainAnimator(all_data, alpha_2)

    if args.save:
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(fps=args.fps, metadata=dict(artist="IsaacLab"))
        animator.anim.save(args.save, writer=writer)
        print(f"Animation saved to {args.save}")
    else:
        animator.show()
