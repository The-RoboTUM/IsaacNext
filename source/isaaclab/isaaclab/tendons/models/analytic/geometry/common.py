# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any


@dataclass
class TendonLengthOutput:
    """Result of one tendon length calculation."""

    delta_l: torch.Tensor
    length: torch.Tensor | None = None
    state: dict[str, torch.Tensor] | None = None
    debug: dict[str, Any] | None = None


@dataclass
class TendonDeltaLengths:
    """All five spring-length deltas used by the analytic tendon model."""

    gst: torch.Tensor
    dft: torch.Tensor
    kft: torch.Tensor
    edt1: torch.Tensor
    edt2: torch.Tensor
    debug: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "gst": self.gst,
            "dft": self.dft,
            "kft": self.kft,
            "edt1": self.edt1,
            "edt2": self.edt2,
        }

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.gst, self.dft, self.kft, self.edt1, self.edt2


@torch.jit.script
def angle_from_sws(a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Angle helper used throughout the tendon geometry equations.

    This is deliberately TorchScript-compatible so the debug and JIT paths use
    the exact same helper implementation.
    """
    x = a - b * torch.cos(theta)
    y = b * torch.sin(theta)
    return torch.atan2(y, x)
