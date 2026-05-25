"""Shared leg-controller interface for BirdBot/Forrest legs.

Controller convention:
    command(t) returns joint position commands in radians for logical DOFs:
    [hip_roll, hip_yaw, hip_flexion, knee_flexion].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np

TWOPI = 2.0 * np.pi

# Logical controller-space DOFs. Controller output order follows this tuple.
DOF_ORDER = ("hip_roll", "hip_yaw", "hip_flexion", "knee_flexion")

# Mapping from logical controller DOFs to FORREST_CFG.actuators keys.
# Keep this out of run.py so run.py never needs concrete robot joint names.
DOF_TO_ACTUATOR_GROUP = {
    "hip_roll": "hip_roll",
    "hip_yaw": "hip_lateral",
    "hip_flexion": "hip_swing",
    "knee_flexion": "knee_flex",
}

# Sign from controller convention to the simulated joint target convention.
DOF_SIGN = {
    "hip_roll": 1.0,
    "hip_yaw": 1.0,
    "hip_flexion": 1.0,
    "knee_flexion": -1.0,
}


def wrap_0_2pi(x: float) -> float:
    y = np.fmod(x, TWOPI)
    if y < 0:
        y += TWOPI
    return float(y)


def theta_warp(phi: float, D: float) -> float:
    r"""Phase warping per duty factor D in (0, 1)."""
    phi = wrap_0_2pi(phi)
    if phi <= TWOPI * D:
        return phi / (2.0 * D)
    return phi / (2.0 * (1.0 - D)) + TWOPI * (1.0 - 2.0 * D) / (2.0 * (1.0 - D))


def _rad_map(values_deg: Mapping[str, float] | None) -> dict[str, float]:
    values_deg = values_deg or {}
    return {dof: float(np.deg2rad(values_deg.get(dof, 0.0))) for dof in DOF_ORDER}


class LegControllerBase(ABC):
    """Base interface for one leg controller."""

    dof_order: Sequence[str] = DOF_ORDER

    @abstractmethod
    def joint(self, dof: str, t: float) -> tuple[float, float]:
        """Return (position, velocity) in rad and rad/s for one logical DOF."""

    def q_serial(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        q = []
        qd = []
        for dof in self.dof_order:
            pos, vel = self.joint(dof, t)
            q.append(pos)
            qd.append(vel)
        return np.asarray(q, dtype=float), np.asarray(qd, dtype=float)

    def command(self, t: float) -> np.ndarray:
        q, _ = self.q_serial(t)
        return q

    def command_velocity(self, t: float) -> np.ndarray:
        _, qd = self.q_serial(t)
        return qd