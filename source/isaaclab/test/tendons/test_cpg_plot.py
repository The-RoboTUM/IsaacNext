# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple CPG plotting test."""

import numpy as np
from pathlib import Path

import pytest

from isaaclab.tendons.controllers.cpg import BirdBotCPGLeg, CPGParams


def test_plot_cpg_angles_two_gait_cycles():
    """Plot all serial joint angles over two gait cycles and save the figure."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = CPGParams(
        f_hz=1.0,
        yaw_A_deg=8.0,
        abd_A_deg=10.0,
    )
    leg = BirdBotCPGLeg(params, include_knee=True)

    period = 1.0 / params.f_hz
    t = np.linspace(0.0, 2.0 * period, 800)

    q = np.array([leg.q_serial(ti)[0] for ti in t])
    q_deg = np.rad2deg(q)

    joint_names = ["yaw", "abduction", "flexion", "knee"]
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for i, name in enumerate(joint_names):
        axs[i].plot(t, q_deg[:, i], linewidth=1.5)
        axs[i].set_ylabel(f"{name} [deg]")
        axs[i].grid(True, alpha=0.3)
    axs[-1].set_xlabel("time [s]")
    fig.suptitle("BirdBot CPG Angles Over Two Gait Cycles")
    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[4] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cpg_angles_two_cycles.png"
    output_path = out_file.as_posix()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    assert out_file.exists()
    assert out_file.stat().st_size > 0
