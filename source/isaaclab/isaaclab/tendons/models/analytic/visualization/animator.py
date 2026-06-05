# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matplotlib animator for tendon actuation/debug data."""

from __future__ import annotations

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from isaaclab.tendons.models.analytic.visualization.context import DEFAULT_ALPHA_2, tc, tids
from isaaclab.tendons.models.analytic.visualization.display import active_color, bool_text, deg_text, table_lines
from isaaclab.tendons.models.analytic.visualization.kinematics import compute_alphas, compute_joint_locations
from isaaclab.tendons.models.analytic.visualization.paths import (
    compute_dft_points,
    compute_edt1_points,
    compute_edt2_points,
    compute_gst_attachment_points,
    compute_kft_points,
)
from isaaclab.tendons.models.analytic.visualization.states import (
    get_dft_state,
    get_edt1_state,
    get_edt2_state,
    get_gst_state,
)
from isaaclab.tendons.models.analytic.visualization.style import log, rest_length_label, set_tendon_active_style
from isaaclab.tendons.models.analytic.visualization.validation import (
    arc_from_3_points,
    compute_h_end_point,
    compute_x_end_point,
    validate_ls,
    validate_xs,
)


class KinematicChainAnimator:
    def __init__(
        self,
        all_data,
        alpha_2=DEFAULT_ALPHA_2,
        *,
        real_time=False,
        data_fps=20.0,
        verbose=False,
        single_plot=False,
        show_debug_geometry=False,
        show_debug_text=False,
        validate_geometry=True,
    ):
        all_thetas = [d["thetas"] for d in all_data]
        self.all_thetas = np.array(all_thetas)
        self.alpha_2 = alpha_2
        self.all_data = all_data
        self.num_frames = len(all_thetas)
        self.current_frame = 0
        self.is_playing = True
        self.real_time = real_time
        self.data_fps = data_fps
        self.verbose = verbose
        self.single_plot = single_plot
        self.show_debug_geometry = show_debug_geometry
        self.show_debug_text = show_debug_text
        self.validate_geometry = validate_geometry
        self.play_start_wall_time = time.perf_counter()
        self.play_start_frame = self.current_frame

        # Set up the figure
        if self.single_plot:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(16, 6.5),
                gridspec_kw={"width_ratios": [5.0, 2.4]},
            )
            shared_ax, shared_info_ax = axes
            self.fig = fig
            self.ax = np.array([[shared_ax, shared_ax], [shared_ax, shared_ax]], dtype=object)
            self.info_ax = np.array([[shared_info_ax, shared_info_ax], [shared_info_ax, shared_info_ax]], dtype=object)
            self._unique_axes = [shared_ax]
            self._info_axes = [shared_info_ax]
        else:
            fig, axes = plt.subplots(
                1,
                8,
                figsize=(32, 6.2),
                gridspec_kw={"width_ratios": [4.0, 1.75, 4.0, 1.75, 4.0, 1.75, 4.0, 1.75]},
            )
            plot_axes = [axes[0], axes[2], axes[4], axes[6]]
            info_axes = [axes[1], axes[3], axes[5], axes[7]]
            self.fig = fig
            # Keep the old 2x2 indexing as references for the future, but display the panels in one horizontal row.
            self.ax = np.array([[plot_axes[0], plot_axes[1]], [plot_axes[2], plot_axes[3]]], dtype=object)
            self.info_ax = np.array([[info_axes[0], info_axes[1]], [info_axes[2], info_axes[3]]], dtype=object)
            self._unique_axes = plot_axes
            self._info_axes = info_axes
        self.n_ax = len(self._unique_axes)
        for ax in self._unique_axes:
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(r"$x\;[\mathrm{m}]$")
            ax.set_ylabel(r"$y\;[\mathrm{m}]$")
        for ax in self._info_axes:
            ax.axis("off")

        # Initialize plot elements
        self.skeleton_lines = []
        self.joints_scatters = []
        self.end_effectors = []
        for ax in self._unique_axes:
            (skeleton_line,) = ax.plot([], [], "b-", linewidth=2, label="Links")
            self.skeleton_lines.append(skeleton_line)
            joints_scatter = ax.scatter([], [], s=5, c="red", zorder=5, label="Joints")
            self.joints_scatters.append(joints_scatter)
            (end_effector,) = ax.plot([], [], "go", markersize=4, label="End effector")
            self.end_effectors.append(end_effector)

        helper_line_alpha = 0.7

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
        (self.gst_upper_tendon_line,) = self.ax[0, 0].plot(
            [], [], "r--", linewidth=1.5, label=rest_length_label("GST upper", tc.upper_gst_length)
        )
        (self.gst_lower_tendon_line,) = self.ax[0, 0].plot(
            [], [], "r--", linewidth=1.5, label=rest_length_label("GST lower", tc.lower_gst_length)
        )

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
        (self.kft_tendon_line,) = self.ax[0, 1].plot(
            [], [], "r--", linewidth=1.5, label=rest_length_label("KFT", tc.kft_length)
        )
        (self.dft_tendon_line,) = self.ax[0, 1].plot(
            [], [], "r--", linewidth=1.5, label=rest_length_label("DFT", tc.dft_length)
        )
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
        (self.edt1_tendon_line,) = self.ax[1, 0].plot(
            [], [], "r--", linewidth=1.5, label=rest_length_label("EDT1", tc.edt1_length)
        )
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
        (self.edt2_tendon_line,) = self.ax[1, 1].plot(
            [], [], "r--", linewidth=1.5, label=rest_length_label("EDT2", tc.edt2_length)
        )
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

        self._debug_geometry_lines = [
            self.gst_x_4prime6_line,
            self.gst_x_4prime7_line,
            self.gst_x_57_line,
            self.gst_h_5_line,
            self.gst_h_6_line,
            self.dft_x_c6_line,
            self.dft_x_57_line,
            self.dft_h_5_line,
            self.dft_h_6_line,
            self.edt1_h_5_line,
            self.edt1_x_c5_line,
            self.edt2_h_5_line,
            self.edt2_h_6_line,
            self.edt2_x_c5_line,
            self.edt2_x_c6_line,
            self.edt2_x_46_line,
            self.edt2_x_5c_line,
        ]
        self._set_debug_geometry_visible(self.show_debug_geometry)

        # Title and info text
        # These live in dedicated right-side axes so they do not overlap the plots.
        if self.single_plot:
            # In single-plot mode all texts live in a single right-side column.
            gst_title_y, gst_info_y, gst_delta_y = 0.98, 0.92, 0.72
            kft_title_y, kft_delta_y, kft_state_y = 0.63, 0.58, 0.53
            dft_title_y, dft_delta_y, dft_state_y = 0.46, 0.41, 0.36
            edt1_title_y, edt1_delta_y, edt1_state_y = 0.28, 0.23, 0.18
            edt2_title_y, edt2_delta_y, edt2_state_y = 0.11, 0.06, 0.01
        else:
            gst_title_y, gst_info_y, gst_delta_y = 0.98, 0.87, 0.50
            kft_title_y, kft_delta_y, kft_state_y = 0.98, 0.88, 0.78
            dft_title_y, dft_delta_y, dft_state_y = 0.56, 0.46, 0.36
            edt1_title_y, edt1_delta_y, edt1_state_y = 0.98, 0.88, 0.76
            edt2_title_y, edt2_delta_y, edt2_state_y = 0.98, 0.88, 0.74

        def right_text(ax, y, text="", *, bold=False, boxed=False):
            return ax.text(
                0.0,
                y,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                fontfamily="DejaVu Sans Mono",
                fontsize=8.0 if not self.single_plot else 7.0,
                fontweight="bold" if bold else "normal",
                linespacing=1.35,
                bbox=(
                    dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.88) if boxed else None
                ),
                clip_on=False,
            )

        self.gst_title = right_text(self.info_ax[0, 0], gst_title_y, "", bold=True)
        self.gst_info_text = right_text(self.info_ax[0, 0], gst_info_y, "", boxed=True)
        # Delta l text in top center with larger font
        self.gst_delta_l_text = right_text(self.info_ax[0, 0], gst_delta_y, "")

        self.kft_title = right_text(self.info_ax[0, 1], kft_title_y, "KFT", bold=True)
        self.kft_delta_l_text = right_text(self.info_ax[0, 1], kft_delta_y, "")
        self.kft_state_text = right_text(self.info_ax[0, 1], kft_state_y, "", boxed=True)
        self.dft_title = right_text(self.info_ax[0, 1], dft_title_y, "DFT", bold=True)
        self.dft_delta_l_text = right_text(self.info_ax[0, 1], dft_delta_y, "")
        self.dft_state_text = right_text(self.info_ax[0, 1], dft_state_y, "", boxed=True)

        self.edt1_title = right_text(self.info_ax[1, 0], edt1_title_y, "EDT1", bold=True)
        self.edt1_delta_l_text = right_text(self.info_ax[1, 0], edt1_delta_y, "")
        self.edt1_state_text = right_text(self.info_ax[1, 0], edt1_state_y, "", boxed=True)

        # EDT2 state text (lower-right)
        self.edt2_title = right_text(self.info_ax[1, 1], edt2_title_y, "EDT2", bold=True)
        # EDT2 delta l text in top center with larger font
        self.edt2_delta_l_text = right_text(self.info_ax[1, 1], edt2_delta_y, "")
        self.edt2_state_text = right_text(self.info_ax[1, 1], edt2_state_y, "", boxed=True)

        # Compute axis limits from all frames
        self._compute_axis_limits()

        # Set up animation
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self._frame_generator,
            interval=20 if self.real_time else max(1, int(1000 / max(self.data_fps, 1e-6))),
            blit=True,
            save_count=self.num_frames,
        )

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        for ax in self._unique_axes:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.48), borderaxespad=0.0)
        if self.single_plot:
            self.fig.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.94, wspace=0.10)
        else:
            self.fig.subplots_adjust(left=0.035, right=0.99, bottom=0.12, top=0.92, wspace=0.35)

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
        for ax in self._unique_axes:
            ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
            ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)

    def _frame_generator(self):
        """Generator that yields frame indices, respecting pause state."""
        while True:
            if self.is_playing:
                if self.real_time:
                    elapsed_s = time.perf_counter() - self.play_start_wall_time
                    target_frame = int(self.play_start_frame + elapsed_s * self.data_fps)
                    self.current_frame = target_frame % self.num_frames
                else:
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
        if self.verbose and frame_idx % 100 == 0:
            log(f"frame {frame_idx}")

        current_data = self.all_data[frame_idx]

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
        ) = compute_gst_attachment_points(self.alpha_2, joints, current_data)
        upper_gst_xs, upper_gst_ys = self.tendon_path(upper_tendon_points, upper_tendon_joints, upper_tendon=True)
        lower_gst_xs, lower_gst_ys = self.tendon_path(
            copy.copy(lower_tendon_points), lower_tendon_joints, upper_tendon=False
        )
        self.gst_upper_tendon_line.set_data(upper_gst_xs, upper_gst_ys)
        self.gst_lower_tendon_line.set_data(lower_gst_xs, lower_gst_ys)

        # validate gst data
        if self.validate_geometry:
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
        dft_h5_b_disengaged = current_data["DFT_h5_B"] > tc.pulley_radii[tids.I_RADIUS_DFT_5].item()
        dft_h5_c_disengaged = current_data["DFT_h5_C"] > tc.pulley_radii[tids.I_RADIUS_DFT_5].item()
        dft_h6_c_disengaged = current_data["DFT_h6_C"] > tc.pulley_radii[tids.I_RADIUS_DFT_6].item()
        dft_h6_d_disengaged = current_data["DFT_h6_D"] > tc.pulley_radii[tids.I_RADIUS_DFT_6].item()
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
        gst_state = get_gst_state(self.all_data[frame_idx])
        dft_state = get_dft_state(self.all_data[frame_idx])
        edt1_state = get_edt1_state(self.all_data[frame_idx])
        edt2_state = get_edt2_state(self.all_data[frame_idx])

        # Update delta l text with conditional coloring
        delta_l = self.all_data[frame_idx]["GST_delta_L_s"]
        gst_active = delta_l <= 0
        delta_l_color = active_color(delta_l)
        self.gst_title.set_text(f"GST — Frame {frame_idx + 1}/{self.num_frames}")
        self.gst_title.set_color("green" if gst_active else "black")
        self.gst_info_text.set_text(
            table_lines(
                [
                    ("mode", status),
                    ("state", gst_state),
                    ("theta3", deg_text(thetas[0])),
                    ("theta4", deg_text(thetas[1])),
                    ("theta5", deg_text(thetas[2])),
                    ("theta6", deg_text(thetas[3])),
                ],
                key_width=8,
            )
        )
        self.gst_delta_l_text.set_text(rf"GST $\Delta L={delta_l * 1000:02.3f}\,\mathrm{{mm}}$")
        self.gst_delta_l_text.set_color(delta_l_color)
        set_tendon_active_style(self.gst_upper_tendon_line, gst_active, "gst_upper", self.single_plot)
        set_tendon_active_style(self.gst_lower_tendon_line, gst_active, "gst_lower", self.single_plot)

        kft_delta_l = self.all_data[frame_idx]["KFT_delta_L_s"]
        kft_active = kft_delta_l <= 0
        kft_delta_l_color = active_color(kft_delta_l)
        self.kft_delta_l_text.set_text(rf"KFT $\Delta L={kft_delta_l * 1000:02.3f}\,\mathrm{{mm}}$")
        self.kft_delta_l_text.set_color(kft_delta_l_color)
        self.kft_state_text.set_text(
            table_lines(
                [
                    ("active", bool_text(kft_active)),
                    ("theta8", deg_text(thetas[4])),
                ],
                key_width=8,
            )
        )
        set_tendon_active_style(self.kft_tendon_line, kft_active, "kft", self.single_plot)

        dft_delta_l = self.all_data[frame_idx]["DFT_delta_L_s"]
        dft_active = dft_delta_l <= 0
        dft_delta_l_color = active_color(dft_delta_l)
        self.dft_delta_l_text.set_text(rf"DFT $\Delta L={dft_delta_l * 1000:02.3f}\,\mathrm{{mm}}$")
        self.dft_delta_l_text.set_color(dft_delta_l_color)
        dft_rows = [
            ("state", dft_state),
            ("active", bool_text(dft_active)),
            ("theta5", deg_text(thetas[2])),
            ("theta6", deg_text(thetas[3])),
        ]
        if self.show_debug_text:
            dft_rows.extend(
                [
                    ("h5_B > r5", bool_text(dft_h5_b_disengaged)),
                    ("h5_C > r5", bool_text(dft_h5_c_disengaged)),
                    ("h6_C > r6", bool_text(dft_h6_c_disengaged)),
                    ("h6_D > r6", bool_text(dft_h6_d_disengaged)),
                ]
            )
        self.dft_state_text.set_text(table_lines(dft_rows, key_width=10))
        set_tendon_active_style(self.dft_tendon_line, dft_active, "dft", self.single_plot)

        self.kft_title.set_text("KFT")
        self.kft_title.set_color("green" if kft_active else "black")
        self.dft_title.set_text("DFT")
        self.dft_title.set_color("green" if dft_active else "black")

        edt1_delta_l = self.all_data[frame_idx]["EDT1_delta_L_s"]
        edt1_active = edt1_delta_l <= 0
        edt1_delta_l_color = active_color(edt1_delta_l)
        self.edt1_title.set_text("EDT1")
        self.edt1_title.set_color("green" if edt1_active else "black")
        self.edt1_delta_l_text.set_text(rf"EDT1 $\Delta L={edt1_delta_l * 1000:02.3f}\,\mathrm{{mm}}$")
        self.edt1_delta_l_text.set_color(edt1_delta_l_color)
        self.edt1_state_text.set_text(
            table_lines(
                [
                    ("state", edt1_state),
                    ("active", bool_text(edt1_active)),
                    ("theta4", deg_text(thetas[1])),
                    ("theta5", deg_text(thetas[2])),
                ],
                key_width=8,
            )
        )
        set_tendon_active_style(self.edt1_tendon_line, edt1_active, "edt1", self.single_plot)

        edt2_delta_l = self.all_data[frame_idx]["EDT2_delta_L_s"]
        edt2_active = edt2_delta_l <= 0
        edt2_delta_l_color = active_color(edt2_delta_l)
        self.edt2_title.set_text("EDT2")
        self.edt2_title.set_color("green" if edt2_active else "black")
        self.edt2_delta_l_text.set_text(rf"EDT2 $\Delta L={edt2_delta_l * 1000:02.3f}\,\mathrm{{mm}}$")
        self.edt2_delta_l_text.set_color(edt2_delta_l_color)
        self.edt2_state_text.set_text(
            table_lines(
                [
                    ("state", edt2_state),
                    ("active", bool_text(edt2_active)),
                    ("theta4", deg_text(thetas[1])),
                    ("theta5", deg_text(thetas[2])),
                    ("theta6", deg_text(thetas[3])),
                ],
                key_width=8,
            )
        )
        set_tendon_active_style(self.edt2_tendon_line, edt2_active, "edt2", self.single_plot)

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
            + [self.gst_title, self.kft_title, self.dft_title, self.edt1_title, self.edt2_title]
            + [self.gst_info_text, self.gst_delta_l_text]
            + [self.kft_delta_l_text, self.kft_state_text, self.dft_delta_l_text, self.dft_state_text]
            + [self.edt1_delta_l_text, self.edt1_state_text]
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

    def _set_debug_geometry_visible(self, visible: bool):
        """Show/hide helper x/h geometry lines without touching tendon paths."""
        self.show_debug_geometry = visible
        for line in getattr(self, "_debug_geometry_lines", []):
            if not hasattr(line, "_tendon_debug_label"):
                line._tendon_debug_label = line.get_label()
            line.set_visible(visible)
            line.set_label(line._tendon_debug_label if visible else "_nolegend_")

    def _toggle_debug_geometry(self):
        """Keyboard helper: toggle the geometry lines that clutter the plots."""
        self._set_debug_geometry_visible(not self.show_debug_geometry)
        self.fig.canvas.draw_idle()

    def _toggle_debug_text(self):
        """Keyboard helper: toggle detailed side-panel conditions."""
        self.show_debug_text = not self.show_debug_text
        self._update(self.current_frame)
        self.fig.canvas.draw_idle()

    def _sync_real_time_clock(self):
        """Keep real-time playback aligned after pause/step/jump events."""
        self.play_start_wall_time = time.perf_counter()
        self.play_start_frame = self.current_frame

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == " ":
            self.is_playing = not self.is_playing
            if self.is_playing:
                self._sync_real_time_clock()
        elif event.key == "right" and not self.is_playing:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self._sync_real_time_clock()
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "left" and not self.is_playing:
            self.current_frame = (self.current_frame - 1) % self.num_frames
            self._sync_real_time_clock()
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "home":
            self.current_frame = 0
            self._sync_real_time_clock()
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "end":
            self.current_frame = self.num_frames - 1
            self._sync_real_time_clock()
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "d":
            self._toggle_debug_geometry()
        elif event.key == "i":
            self._toggle_debug_text()

    def show(self):
        """Display the animation."""
        plt.show()
