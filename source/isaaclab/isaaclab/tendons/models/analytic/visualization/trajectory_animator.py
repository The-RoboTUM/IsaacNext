# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory-only animator for recordings without tendon debug frames."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from isaaclab.tendons.models.analytic.visualization.context import DEFAULT_ALPHA_2
from isaaclab.tendons.models.analytic.visualization.kinematics import compute_alphas, compute_joint_locations
from isaaclab.tendons.models.analytic.visualization.style import log


class TrajectoryOnlyAnimator:
    """Animate the recorded leg chain without drawing tendon paths."""

    def __init__(
        self,
        all_data,
        alpha_2=DEFAULT_ALPHA_2,
        *,
        real_time=False,
        data_fps=20.0,
        verbose=False,
    ):
        self.all_thetas = np.array([data["thetas"] for data in all_data])
        self.all_data = all_data
        self.alpha_2 = alpha_2
        self.num_frames = len(all_data)
        self.current_frame = 0
        self.is_playing = True
        self.real_time = real_time
        self.data_fps = data_fps
        self.verbose = verbose
        self.play_start_wall_time = time.perf_counter()
        self.play_start_frame = self.current_frame

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel(r"$x\;[\mathrm{m}]$")
        self.ax.set_ylabel(r"$y\;[\mathrm{m}]$")
        (self.skeleton_line,) = self.ax.plot([], [], "b-", linewidth=2, label="Links")
        self.joints_scatter = self.ax.scatter([], [], s=18, c="red", zorder=5, label="Joints")
        (self.end_effector,) = self.ax.plot([], [], "go", markersize=5, label="End effector")
        self.time_text = self.ax.text(0.02, 0.96, "", transform=self.ax.transAxes, va="top")

        self._compute_axis_limits()
        self.ax.legend(loc="upper right")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=self._frame_generator,
            interval=20 if self.real_time else max(1, int(1000 / max(self.data_fps, 1e-6))),
            blit=True,
            save_count=self.num_frames,
        )

    def _compute_axis_limits(self) -> None:
        all_x, all_y = [], []
        for thetas in self.all_thetas:
            joints = compute_joint_locations(compute_alphas(self.alpha_2, thetas))
            all_x.extend(float(joint[0]) for joint in joints)
            all_y.extend(float(joint[1]) for joint in joints)
        margin = 0.15
        x_range = max(max(all_x) - min(all_x), 1.0e-6)
        y_range = max(max(all_y) - min(all_y), 1.0e-6)
        self.ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
        self.ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)

    def _frame_generator(self):
        while True:
            if self.is_playing:
                if self.real_time:
                    elapsed_s = time.perf_counter() - self.play_start_wall_time
                    self.current_frame = int(self.play_start_frame + elapsed_s * self.data_fps) % self.num_frames
                else:
                    self.current_frame = (self.current_frame + 1) % self.num_frames
            yield self.current_frame

    def _update(self, frame_idx):
        if self.verbose and frame_idx % 100 == 0:
            log(f"frame {frame_idx}")
        frame = self.all_data[frame_idx]
        joints = compute_joint_locations(compute_alphas(self.alpha_2, self.all_thetas[frame_idx]))
        xs = [joint[0] for joint in joints]
        ys = [joint[1] for joint in joints]
        self.skeleton_line.set_data(xs, ys)
        self.joints_scatter.set_offsets(np.c_[xs[:-1], ys[:-1]])
        self.end_effector.set_data([xs[-1]], [ys[-1]])
        self.time_text.set_text(f"frame {frame_idx + 1}/{self.num_frames}  t={frame.get('sim_time', 0.0):.3f}s")
        return [self.skeleton_line, self.joints_scatter, self.end_effector, self.time_text]

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.is_playing = not self.is_playing
            self.play_start_wall_time = time.perf_counter()
            self.play_start_frame = self.current_frame
        elif event.key == "right":
            self.is_playing = False
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "left":
            self.is_playing = False
            self.current_frame = (self.current_frame - 1) % self.num_frames
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "home":
            self.is_playing = False
            self.current_frame = 0
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()
        elif event.key == "end":
            self.is_playing = False
            self.current_frame = self.num_frames - 1
            self._update(self.current_frame)
            self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()
