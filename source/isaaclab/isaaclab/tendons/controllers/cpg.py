# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPG-based controller for BirdBot/Forrest legs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaaclab.tendons.controllers.base import DOF_ORDER, TWOPI, LegControllerBase, theta_warp, wrap_0_2pi


@dataclass
class CPGParams:
    f_hz: float = 1.0
    D: float = 0.6
    A_h_deg: float = 32.0
    O_h_deg: float = 22.0
    A_k_deg: float = 120.0
    S_f: float = 0.0
    S_e: float = 0.22
    phi0: float = 0.0
    yaw_A_deg: float = 0.0
    yaw_O_deg: float = 0.0
    roll_A_deg: float = 0.0
    roll_O_deg: float = 0.0


class BirdBotCPGLeg(LegControllerBase):
    """CPG-based leg controller using the common controller interface."""

    def __init__(self, params: CPGParams, include_knee: bool = True):
        self.p = params
        self.include_knee = include_knee
        self.A_h = np.deg2rad(self.p.A_h_deg)
        self.O_h = np.deg2rad(self.p.O_h_deg)
        self.A_k = np.deg2rad(self.p.A_k_deg)
        self.yaw_A = np.deg2rad(self.p.yaw_A_deg)
        self.yaw_O = np.deg2rad(self.p.yaw_O_deg)
        self.roll_A = np.deg2rad(self.p.roll_A_deg)
        self.roll_O = np.deg2rad(self.p.roll_O_deg)
        self.omega = TWOPI * self.p.f_hz

    def _phi(self, t: float) -> float:
        return self.omega * t + self.p.phi0

    def joint(self, dof: str, t: float) -> tuple[float, float]:
        if dof == "hip_roll":
            return self.hip_roll(t)
        if dof == "hip_yaw":
            return self.hip_yaw(t)
        if dof == "hip_flexion":
            return self.hip_flex(t)
        if dof == "knee_flexion":
            return self.knee(t)
        raise KeyError(f"Unknown controller DOF: {dof}")

    def hip_flex(self, t: float) -> tuple[float, float]:
        phi = self._phi(t)
        theta = theta_warp(phi, self.p.D)

        if wrap_0_2pi(phi) <= TWOPI * self.p.D:
            theta_dot = self.omega / (2.0 * self.p.D)
        else:
            theta_dot = self.omega / (2.0 * (1.0 - self.p.D))

        q = self.A_h * np.sin(theta + np.pi / 2.0) + self.O_h
        qd = self.A_h * np.cos(theta + np.pi / 2.0) * theta_dot
        return float(q), float(qd)

    def hip_roll(self, t: float) -> tuple[float, float]:
        if self.roll_A == 0.0 and self.roll_O == 0.0:
            return 0.0, 0.0

        phi = self._phi(t)
        q = self.roll_A * np.sin(phi) + self.roll_O
        qd = self.roll_A * np.cos(phi) * self.omega
        return float(q), float(qd)

    # Backward-compatible alias for old abduction naming.
    hip_abd = hip_roll

    def hip_yaw(self, t: float) -> tuple[float, float]:
        if self.yaw_A == 0.0 and self.yaw_O == 0.0:
            return 0.0, 0.0

        phi = self._phi(t)
        q = self.yaw_A * np.sin(phi) + self.yaw_O
        qd = self.yaw_A * np.cos(phi) * self.omega
        return float(q), float(qd)

    def knee(self, t: float) -> tuple[float, float]:
        """Knee flexion command, positive in controller convention."""
        if not self.include_knee:
            return 0.0, 0.0

        phi = wrap_0_2pi(self._phi(t))
        phi_on_lo = TWOPI * self.p.D + TWOPI * self.p.S_f
        phi_on_hi = TWOPI - TWOPI * self.p.S_e

        if not (phi_on_lo <= phi <= phi_on_hi):
            return 0.0, 0.0

        denom = max(phi_on_hi - phi_on_lo, 1e-8)
        swing_phase = (phi - phi_on_lo) / denom

        q = self.A_k * np.sin(np.pi * swing_phase)
        qd = self.A_k * np.cos(np.pi * swing_phase) * np.pi / denom * self.omega
        return float(q), float(qd)


def make_birdbot_cpg_controller(params: CPGParams, map_matrix=None, include_knee=False):
    """Backward-compatible actuator-wrapper factory."""
    leg = BirdBotCPGLeg(params, include_knee=include_knee)
    M = np.eye(len(DOF_ORDER)) if map_matrix is None else np.asarray(map_matrix, dtype=float)

    def make_fun(i: int, vel: bool = False):
        def f(t: float):
            q = leg.command_velocity(t) if vel else leg.command(t)
            return float(M[i, : len(q)] @ q)

        return f

    qd_funs = [make_fun(i, vel=False) for i in range(M.shape[0])]
    qd_dot_funs = [make_fun(i, vel=True) for i in range(M.shape[0])]
    return qd_funs, qd_dot_funs, leg
