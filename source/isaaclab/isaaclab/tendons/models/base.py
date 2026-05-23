from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


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
