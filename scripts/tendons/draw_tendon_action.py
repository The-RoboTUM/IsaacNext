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
from matplotlib.path import Path
from matplotlib import patches
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from isaaclab.tendons.constants import (
    TendonData,
    tids,
    TendonConstants,
    dummy_randomization,
)

tc = TendonConstants()

all_data = []
with open("outputs/gst_data_left.jsonl", "r") as f:
    for line in f:
        all_data.append(json.loads(line))


# draw the leg, starting at joint 2
alpha_2 = np.deg2rad(300)
tendon_data = TendonData(1, dummy_randomization)


def compute_alphas(alpha_2, thetas):
    theta_3, theta_4, theta_5, theta_6 = thetas
    alpha_3 = np.pi + alpha_2 - theta_3
    alpha_4 = alpha_3 + theta_4 - np.pi
    alpha_5 = alpha_4 + theta_5 - np.pi
    alpha_6 = alpha_5 + theta_6 - np.pi
    return [alpha_2, alpha_3, alpha_4, alpha_5, alpha_6]


def rotate_by(angle, vector):
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s], [s, c]]) @ vector


def compute_joint_locations(alphas):
    alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = alphas
    j2 = np.zeros(2)
    j3 = j2 + rotate_by(
        alpha_2, np.array([tc.link_lengths[tids.I_LINK_23].item(), 0.0])
    )
    j4 = j3 + rotate_by(
        alpha_3, np.array([tc.link_lengths[tids.I_LINK_34].item(), 0.0])
    )
    j5 = j4 + rotate_by(
        alpha_4, np.array([tc.link_lengths[tids.I_LINK_4prime5].item(), 0.0])
    )
    j6 = j5 + rotate_by(
        alpha_5, np.array([tc.link_lengths[tids.I_LINK_56].item(), 0.0])
    )
    p7 = j6 + rotate_by(
        alpha_6, np.array([tc.link_lengths[tids.I_LINK_67].item(), 0.0])
    )

    return [j2, j3, j4, j5, j6, p7]


# takes alpha_2, beta_2, assumes starting point at r(49.5°) @ [0.059407 0]
def compute_tendon_attachment_points(alpha_2, joint_locations, data):
    [j2, j3, j4, j5, j6, _] = joint_locations
    state = data["state"]
    q3 = data["qs"][0]
    q4 = data["q4"]
    q4prime = data["q4prime"]
    q5 = data["qs"][-2]
    q6 = data["qs"][-1]
    l4prime6 = data["l_4prime6"]
    l4prime7 = data["l_4prime7"]
    l57 = data["l_57"]
    q6_B = data["q6_B"]
    q5_D = data["q5_D"]
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
        np.array([tendon_data.tendon_section_lengths[0, tids.I_LINK_23].item(), 0.0]),
    )
    p3_o = rotate_by(-q3, p3_i - j3) + j3
    direction_angle -= q3
    p4_i = p3_o + rotate_by(
        direction_angle,
        np.array([tendon_data.tendon_section_lengths[0, tids.I_LINK_34].item(), 0.0]),
    )
    p4_o = (
        rotate_by(
            q4
            + np.deg2rad(
                30
            ),  # note: added extra angle to avoid issues with negative values
            p4_i - j4,
        )
        + j4
    )
    direction_angle += q4
    radius_4 = np.linalg.norm(p4_o - j4)
    assert (
        abs(radius_4 - tendon_data.pulley_radii[0, tids.I_RADIUS_4].item()) < 0.001
    ), f"Expected radius at 4 {tendon_data.pulley_radii[0, tids.I_RADIUS_4].item()}, got {radius_4}"
    p4prime_i = (
        rotate_by(
            np.deg2rad(-30),  # note: to compensate for extended upper tendon drawing
            (
                (p4_o - j4)
                * (
                    radius_4
                    - tendon_data.pulley_radii[0, tids.I_RADIUS_4].item()
                    + tendon_data.pulley_radii[0, tids.I_RADIUS_4prime].item()
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

    lower_tendon_points = [p4prime_i, p4prime_o]
    lower_tendon_joints = [j4]
    if state[-1] == "a":
        p5_i = p4prime_o + rotate_by(
            direction_angle,
            np.array(
                [tendon_data.tendon_section_lengths[0, tids.I_LINK_4prime5].item(), 0.0]
            ),
        )
        p5_o = rotate_by(q5, p5_i - j5) + j5
        direction_angle += q5
        p6_i = p5_o + rotate_by(
            direction_angle,
            np.array(
                [tendon_data.tendon_section_lengths[0, tids.I_LINK_56].item(), 0.0]
            ),
        )
        p6_o = rotate_by(q6, p6_i - j6) + j6
        direction_angle += q6
        p7 = p6_o + rotate_by(
            direction_angle,
            np.array(
                [tendon_data.tendon_section_lengths[0, tids.I_LINK_67].item(), 0.0]
            ),
        )
        lower_tendon_points.extend([p5_i, p5_o, p6_i, p6_o, p7])
        lower_tendon_joints.extend([j5, j6])
    elif state[-1] == "b":
        p6_i = p4prime_o + rotate_by(direction_angle, np.array([l4prime6, 0.0]))
        p6_o = rotate_by(q6_B, p6_i - j6) + j6
        direction_angle += q6_B
        p7 = p6_o + rotate_by(
            direction_angle,
            np.array(
                [tendon_data.tendon_section_lengths[0, tids.I_LINK_67].item(), 0.0]
            ),
        )
        lower_tendon_points.extend([p6_i, p6_o, p7])
        lower_tendon_joints.extend([j6])

    elif state[-1] == "c":
        p7 = p4prime_o + rotate_by(direction_angle, np.array([l4prime7, 0.0]))
        lower_tendon_points.extend([p7])

    elif state[-1] == "d":
        p5_i = p4prime_o + rotate_by(
            direction_angle,
            np.array(
                [tendon_data.tendon_section_lengths[0, tids.I_LINK_4prime5].item(), 0.0]
            ),
        )
        p5_o = rotate_by(q5_D, p5_i - j5) + j5
        direction_angle += q5_D
        p7 = p5_o + rotate_by(direction_angle, np.array([l57, 0.0]))
        lower_tendon_points.extend([p5_i, p5_o, p7])
        lower_tendon_joints.extend([j5])
    else:
        raise ValueError(f"state {state} not recognized")

    return (
        upper_tendon_points,
        upper_tendon_joints,
        lower_tendon_points,
        lower_tendon_joints,
    )


# for state A: none, state B: x4'6 with h5B, state C: x4'6, x4'7, h5C, h6C, state D: x57, h6D; draw using phis
def validate_xs(joint_locations, x4prime6, x4prime7, x57):
    [_j2, _j3, j4, j5, j6, p7] = joint_locations
    assert (
        abs(np.linalg.norm(j6 - j4) - x4prime6) < 0.001
    ), f"expected x4'6={np.linalg.norm(j6 - j4)}, got {x4prime6}"
    assert (
        abs(np.linalg.norm(p7 - j4) - x4prime7) < 0.001
    ), f"expected x4'7={np.linalg.norm(p7 - j4)}, got {x4prime7}"
    assert (
        abs(np.linalg.norm(p7 - j5) - x57) < 0.001
    ), f"expected x57={np.linalg.norm(p7 - j5)}, got {x57}"


def validate_ls(joint_locations, l4prime6, l4prime7, l57):
    [_j2, _j3, j4, j5, j6, p7] = joint_locations
    x46_inf = np.linalg.norm(j6 - j4)
    l46_inf = np.sqrt(
        x46_inf**2
        - (
            (tc.pulley_radii[tids.I_RADIUS_4prime] - tc.pulley_radii[tids.I_RADIUS_6])
            ** 2
        ).item()
    )
    assert abs(l46_inf - l4prime6) < 0.001, f"Expected l4'6={l46_inf}, got {l4prime6}"

    x47_inf = np.linalg.norm(p7 - j4)
    l47_inf = np.sqrt(x47_inf**2 - (tc.pulley_radii[tids.I_RADIUS_4prime] ** 2).item())
    assert abs(l47_inf - l4prime7) < 0.001, f"Expected l4'7={l47_inf}, got {l4prime7}"

    x57_inf = np.linalg.norm(p7 - j5)
    l57_inf = np.sqrt(x57_inf**2 - (tc.pulley_radii[tids.I_RADIUS_5] ** 2).item())
    assert abs(l57_inf - l57) < 0.001, f"Expected l57={l57_inf}, got {l57}"


def arc_from_3_points(c, p1, p2, ccw=True, tol=1e-4):
    d1 = p1 - c
    d2 = p2 - c
    assert np.isclose(
        np.linalg.norm(d1), np.linalg.norm(d2), atol=tol
    ), "start and end points should have same radius"
    r = np.linalg.norm(d1)
    # assert np.isclose(d1, r, atol=tol), f"Start point not on circle: {d1} != {r}"
    # assert np.isclose(d2, r, atol=tol), f"End point not on circle: {d2} != {r}"
    theta_1 = np.arctan2(d1[1], d1[0])
    theta_2 = np.arctan2(d2[1], d2[0])
    if theta_2 < theta_1 and ccw:
        theta_2 += 2 * np.pi
    if theta_1 < theta_2 and not ccw:
        theta_1 += 2 * np.pi

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
    assert abs(dist - height) < 0.001, f"Expected height {height}, got {dist}"
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
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

        # Initialize plot elements
        (self.skeleton_line,) = self.ax.plot([], [], "b-", linewidth=2, label="Links")
        self.joints_scatter = self.ax.scatter(
            [], [], s=25, c="red", zorder=5, label="Joints"
        )
        (self.end_effector,) = self.ax.plot(
            [], [], "go", markersize=5, label="End effector"
        )

        (self.upper_tendon_line,) = self.ax.plot(
            [], [], "r-", linewidth=2, label="Upper Tendon"
        )
        (self.lower_tendon_line,) = self.ax.plot(
            [], [], "y-", linewidth=2, label="Lower Tendon"
        )
        helper_line_alpha = 0.7
        (self.x_4prime6_line,) = self.ax.plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x4'6",
            alpha=helper_line_alpha,
        )
        (self.x_4prime7_line,) = self.ax.plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x4'7",
            alpha=helper_line_alpha,
        )
        (self.x_57_line,) = self.ax.plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="x57",
            alpha=helper_line_alpha,
        )
        (self.h_5_line,) = self.ax.plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h5",
            alpha=helper_line_alpha,
        )
        (self.h_6_line,) = self.ax.plot(
            [],
            [],
            linestyle="solid",
            color="grey",
            linewidth=2,
            label="h6",
            alpha=helper_line_alpha,
        )

        # Pulley circles (will be updated each frame)
        self.pulley_circles = []

        # Title and info text
        self.title = self.ax.set_title("")
        self.info_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Compute axis limits from all frames
        self._compute_axis_limits()

        # Set up animation
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self._frame_generator,
            interval=50,
            blit=False,
            save_count=self.num_frames,
        )

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax.legend(loc="upper right")

    def _compute_axis_limits(self):
        """Pre-compute axis limits based on all frames."""
        all_x, all_y = [], []
        for thetas in self.all_thetas:
            alphas = compute_alphas(self.alpha_2, thetas)
            joints = compute_joint_locations(alphas)
            for j in joints:
                all_x.append(j[0])
                all_y.append(j[1])

        margin = 0.05
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        self.ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
        self.ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)

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

    def _update(self, frame_idx):
        """Update function for animation."""
        if frame_idx % 100 == 0:
            print(f"frame {frame_idx}")
        current_data = all_data[frame_idx]
        state = current_data["state"][-1]
        thetas = self.all_thetas[frame_idx]
        alphas = compute_alphas(self.alpha_2, thetas)
        joints = compute_joint_locations(alphas)
        (
            upper_tendon_points,
            upper_tendon_joints,
            lower_tendon_points,
            lower_tendon_joints,
        ) = compute_tendon_attachment_points(alpha_2, joints, current_data)
        j2, j3, j4, j5, j6, p7 = joints

        # Update skeleton line (connecting all joints)
        x_coords = [j2[0], j3[0], j4[0], j5[0], j6[0], p7[0]]
        y_coords = [j2[1], j3[1], j4[1], j5[1], j6[1], p7[1]]
        self.skeleton_line.set_data(x_coords, y_coords)

        # Update joint positions
        joint_x = [j[0] for j in joints[:-1]]  # Exclude end effector
        joint_y = [j[1] for j in joints[:-1]]
        self.joints_scatter.set_offsets(np.c_[joint_x, joint_y])

        # Update end effector
        self.end_effector.set_data([p7[0]], [p7[1]])

        # Remove old pulley circles
        for circle in self.pulley_circles:
            circle.remove()
        self.pulley_circles = []

        # Draw pulley circles
        pulley_configs = [
            (j3, tc.pulley_radii[tids.I_RADIUS_3].item(), "orange"),
            (j4, tc.pulley_radii[tids.I_RADIUS_4].item(), "purple"),
            (j4, tc.pulley_radii[tids.I_RADIUS_4prime].item(), "magenta"),
            (j5, tc.pulley_radii[tids.I_RADIUS_5].item(), "cyan"),
            (j6, tc.pulley_radii[tids.I_RADIUS_6].item(), "yellow"),
        ]
        for center, radius, color in pulley_configs:
            circle = Circle(
                center, radius, fill=False, edgecolor=color, linewidth=1.5, alpha=0.7
            )
            self.ax.add_patch(circle)
            self.pulley_circles.append(circle)

        # Draw tendons
        upper_gst_xs, upper_gst_ys = self.tendon_path(
            upper_tendon_points, upper_tendon_joints, upper_tendon=True
        )
        lower_gst_xs, lower_gst_ys = self.tendon_path(
            copy.copy(lower_tendon_points), lower_tendon_joints, upper_tendon=False
        )
        self.upper_tendon_line.set_data(upper_gst_xs, upper_gst_ys)
        self.lower_tendon_line.set_data(lower_gst_xs, lower_gst_ys)

        # depending on the state, draw xs and hs
        validate_xs(
            joints,
            current_data["x_4prime6"],
            current_data["x_4prime7"],
            current_data["x_57"],
        )
        validate_ls(
            joints,
            current_data["l_4prime6"],
            current_data["l_4prime7"],
            current_data["l_57"],
        )
        self.x_4prime6_line.set_data([], [])
        self.x_4prime7_line.set_data([], [])
        self.x_57_line.set_data([], [])
        self.h_5_line.set_data([], [])
        self.h_6_line.set_data([], [])
        if state == "b":
            x_4prime6_end_point = compute_x_end_point(
                j4,
                j5,
                -current_data["phi_4prime_a"],
                current_data["x_4prime6"],
                goal_point=j6,
            )  # rotate 4'5 by -phi_4'a and scale
            self.x_4prime6_line.set_data(
                [j4[0], x_4prime6_end_point[0]], [j4[1], x_4prime6_end_point[1]]
            )
            h5_end_point = compute_h_end_point(
                lower_tendon_points[1], lower_tendon_points[2], j5, current_data["h5_B"]
            )  # project j5 onto according line segment, assert distance is equal to h5B
            self.h_5_line.set_data([j5[0], h5_end_point[0]], [j5[1], h5_end_point[1]])
        elif state == "c":
            x_4prime6_end_point = compute_x_end_point(
                j4,
                j5,
                -current_data["phi_4prime_a"],
                current_data["x_4prime6"],
                goal_point=j6,
            )  # rotate 4'5 by -phi_4'a and scale
            self.x_4prime6_line.set_data(
                [j4[0], x_4prime6_end_point[0]], [j4[1], x_4prime6_end_point[1]]
            )
            h5_end_point = compute_h_end_point(
                lower_tendon_points[1], lower_tendon_points[2], j5, current_data["h5_C"]
            )  # project j5 onto according line segment, assert distance is equal to h5C
            self.h_5_line.set_data([j5[0], h5_end_point[0]], [j5[1], h5_end_point[1]])

            x_4prime7_end_point = compute_x_end_point(
                j4,
                j5,
                -current_data["phi_4prime_a"] - current_data["phi_4prime_d"],
                current_data["x_4prime7"],
                goal_point=p7,
            )  # rotate 4'5 by -phi_4'a-phi_4'd and scale
            self.x_4prime7_line.set_data(
                [j4[0], x_4prime7_end_point[0]], [j4[1], x_4prime7_end_point[1]]
            )
            h6_end_point = compute_h_end_point(
                lower_tendon_points[1], lower_tendon_points[2], j6, current_data["h6_C"]
            )  # project j6 onto according line segment, assert distance is equal to h6C
            self.h_6_line.set_data([j6[0], h6_end_point[0]], [j6[1], h6_end_point[1]])

        elif state == "d":
            x_57_end_point = compute_x_end_point(
                j5, j6, -current_data["phi_5_a"], current_data["x_57"], goal_point=p7
            )  # rotate 56 by -phi_5a and scale
            self.x_57_line.set_data(
                [j5[0], x_57_end_point[0]], [j5[1], x_57_end_point[1]]
            )
            h6_end_point = compute_h_end_point(
                lower_tendon_points[-2],
                lower_tendon_points[-1],
                j6,
                current_data["h6_D"],
            )  # project j6 onto according line segment, assert distance is equal to h6B
            self.h_6_line.set_data([j6[0], h6_end_point[0]], [j6[1], h6_end_point[1]])

        # Update title and info
        status = "▶ Playing" if self.is_playing else "⏸ Paused"
        state = all_data[frame_idx]["state"]
        self.title.set_text(
            f"Kinematic Chain Animation - Frame {frame_idx + 1}/{self.num_frames}"
        )
        self.info_text.set_text(
            f"{status}\n"
            f"State: {state}\n"
            f"$\\Delta l$: {all_data[frame_idx]['delta_l'] * 100:.3f} cm\n"
            f"θ₃: {np.rad2deg(thetas[0]):.1f}°\n"
            f"θ₄: {np.rad2deg(thetas[1]):.1f}°\n"
            f"θ₅: {np.rad2deg(thetas[2]):.1f}°\n"
            f"θ₆: {np.rad2deg(thetas[3]):.1f}°"
        )

        return self.skeleton_line, self.joints_scatter, self.end_effector

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
        default=15,
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
