"""Validation and drawing-geometry primitives for the tendon visualizer."""

from __future__ import annotations

import numpy as np

from isaaclab.tendons.models.analytic.visualization.context import tc, tids
from isaaclab.tendons.models.analytic.visualization.kinematics import rotate_by


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
    # if theta_1 < theta_2 and not ccw and q_positive:  # todo: check here
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
