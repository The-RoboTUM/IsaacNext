"""Simple sinusoidal controller for BirdBot/Forrest legs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from isaaclab.tendons.controllers.base import DOF_ORDER, TWOPI, LegControllerBase, _rad_map

@dataclass
class SinusoidalParams:
    f_hz: float = 1.0
    phi0: float = 0.0
    amplitude_deg: Mapping[str, float] = field(default_factory=dict)
    offset_deg: Mapping[str, float] = field(default_factory=dict)
    phase_rad: Mapping[str, float] = field(default_factory=dict)


class SinusoidalLegController(LegControllerBase):
    """Simple per-DOF sinusoidal controller."""

    def __init__(self, params: SinusoidalParams):
        self.p = params
        self.omega = TWOPI * self.p.f_hz
        self.amplitude = _rad_map(self.p.amplitude_deg)
        self.offset = _rad_map(self.p.offset_deg)
        self.phase = {dof: float(self.p.phase_rad.get(dof, 0.0)) for dof in DOF_ORDER}

    def joint(self, dof: str, t: float) -> tuple[float, float]:
        if dof not in DOF_ORDER:
            raise KeyError(f"Unknown controller DOF: {dof}")

        phi = self.omega * t + self.p.phi0 + self.phase[dof]
        q = self.amplitude[dof] * np.sin(phi) + self.offset[dof]
        qd = self.amplitude[dof] * np.cos(phi) * self.omega
        return float(q), float(qd)