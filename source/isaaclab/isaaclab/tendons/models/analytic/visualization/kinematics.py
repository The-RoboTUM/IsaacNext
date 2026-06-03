"""2-D kinematic helpers used by the tendon visualizer."""

from __future__ import annotations

import numpy as np

from isaaclab.tendons.models.analytic.visualization.context import td, tids


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
        np.array([td.link_lengths[0, tids.I_LINK_23].item(), 0.0]),
    )
    j4 = j3 + rotate_by(alpha_3, np.array([td.link_lengths[0, tids.I_LINK_34].item(), 0.0]))
    j5 = j4 + rotate_by(alpha_4, np.array([td.link_lengths[0, tids.I_LINK_4prime5].item(), 0.0]))
    j6 = j5 + rotate_by(alpha_5, np.array([td.link_lengths[0, tids.I_LINK_56].item(), 0.0]))
    p7 = j6 + rotate_by(alpha_6, np.array([td.link_lengths[0, tids.I_LINK_67].item(), 0.0]))

    return [j2, j3, j4, j5, j6, p7]
