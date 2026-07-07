# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flattened parameter-space utilities for PSO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from isaaclab.pso.config import ParameterSpec


class ParameterSpace:
    """Map named scalar parameters to normalized PSO tensors."""

    def __init__(self, specs: list[ParameterSpec], *, device: str | torch.device):
        if not specs:
            raise ValueError("PSO requires at least one optimized parameter.")
        names = [spec.name for spec in specs]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"Duplicate PSO parameter names: {duplicates}")

        self.specs = specs
        self.names = tuple(names)
        self.device = torch.device(device)
        self.low = torch.tensor([spec.min for spec in specs], dtype=torch.float32, device=self.device)
        self.high = torch.tensor([spec.max for spec in specs], dtype=torch.float32, device=self.device)
        if torch.any(self.high <= self.low):
            bad = [spec.name for spec in specs if spec.max <= spec.min]
            raise ValueError(f"PSO parameter max must be greater than min: {bad}")

    @property
    def dim(self) -> int:
        return len(self.specs)

    def initial_normalized(self) -> torch.Tensor:
        values = []
        for spec in self.specs:
            if spec.initial is None:
                values.append(0.5)
            else:
                values.append((float(spec.initial) - float(spec.min)) / (float(spec.max) - float(spec.min)))
        return torch.tensor(values, dtype=torch.float32, device=self.device).clamp(0.0, 1.0)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.low + normalized.clamp(0.0, 1.0) * (self.high - self.low)

    def normalize(self, physical: torch.Tensor) -> torch.Tensor:
        return ((physical - self.low) / (self.high - self.low)).clamp(0.0, 1.0)

    def batch_to_named_tensors(self, physical: torch.Tensor) -> dict[str, torch.Tensor]:
        if physical.ndim != 2 or physical.shape[1] != self.dim:
            raise ValueError(f"Expected physical parameter tensor shape (N, {self.dim}), got {tuple(physical.shape)}")
        return {name: physical[:, index] for index, name in enumerate(self.names)}

    def vector_to_dict(self, physical: torch.Tensor) -> dict[str, float]:
        if physical.ndim != 1 or physical.shape[0] != self.dim:
            raise ValueError(f"Expected physical vector shape ({self.dim},), got {tuple(physical.shape)}")
        values = physical.detach().cpu().tolist()
        return {name: float(values[index]) for index, name in enumerate(self.names)}

    def export_forrest_yaml(
        self,
        path: str | Path,
        physical: torch.Tensor,
        *,
        includes: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Write a Forrest-compatible override YAML containing optimized values."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = self.vector_to_dict(physical)
        data: dict[str, Any] = {}
        if includes:
            data["includes"] = [_resolve_include_path(include) for include in includes]
        for name, value in values.items():
            _set_nested(data, name.split("."), value)
        run_data = data.setdefault("run", {})
        if any(name.startswith("run.cpg_oscillator.") for name in values):
            run_data["controller"] = "cpg_oscillator"
        elif any(name.startswith("run.cpg.") for name in values):
            run_data["controller"] = "cpg"
        elif any(name.startswith("run.sinusoidal.") for name in values):
            run_data["controller"] = "sin"

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False)
        return data


def _set_nested(data: dict[str, Any], keys: list[str], value: float) -> None:
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        if not isinstance(current[key], dict):
            raise ValueError(f"Cannot set nested parameter path through non-dict key: {'.'.join(keys)}")
        current = current[key]
    current[keys[-1]] = float(value)


def _resolve_include_path(path: str | Path) -> str:
    include_path = Path(path).expanduser()
    if not include_path.is_absolute():
        include_path = include_path.resolve()
    return include_path.as_posix()
