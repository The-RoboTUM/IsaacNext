# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TendonModelOutput:
    energy: torch.Tensor
    per_tendon_energy: dict[str, torch.Tensor] | None = None
    delta_lengths: dict[str, torch.Tensor] | None = None
    debug: dict[str, Any] | None = None


class TendonEnergyModel(ABC):
    @abstractmethod
    def energy(self, joint_angles: torch.Tensor, *, debug: bool = False) -> TendonModelOutput:
        raise NotImplementedError
