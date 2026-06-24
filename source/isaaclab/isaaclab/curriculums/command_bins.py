# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Binned velocity-command curriculum shared by RL and PSO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CommandBinCurriculumParameters:
    """Configuration for progressively unlocking velocity-command bins."""

    enabled: bool = True
    bins: list[dict[str, list[float] | tuple[float, float]]] | None = None
    include_stand_bin: bool = True
    lin_vel_x_min: float = 0.0
    lin_vel_x_max: float = 2.0
    lin_vel_x_bin_width: float = 0.1
    lin_vel_y: float = 0.0
    ang_vel_z: float = 0.0
    initial_unlocked_bin: int = 0
    successes_to_unlock: int = 64
    min_attempts_to_unlock: int = 128
    min_success_rate_to_unlock: float = 0.5
    max_attempts_to_track: int = 0
    success_velocity_tolerance: float = 0.25
    success_yaw_rate_tolerance: float = 0.4
    success_min_survival_fraction: float = 0.98
    sample_only_frontier: bool = False
    reset_counts_on_unlock: bool = False
    command_name: str = "base_velocity"


def generated_bins_from_params(params: CommandBinCurriculumParameters) -> list[dict[str, tuple[float, float]]]:
    """Build +x-only velocity bins from scalar curriculum parameters."""

    x_min = float(params.lin_vel_x_min)
    x_max = float(params.lin_vel_x_max)
    width = float(params.lin_vel_x_bin_width)
    if x_max < x_min:
        raise ValueError("Command curriculum requires lin_vel_x_max >= lin_vel_x_min.")
    if width <= 0.0:
        raise ValueError("Command curriculum requires lin_vel_x_bin_width > 0.")

    bins: list[dict[str, tuple[float, float]]] = []
    y = float(params.lin_vel_y)
    yaw = float(params.ang_vel_z)
    if bool(params.include_stand_bin):
        bins.append({"lin_vel_x": (0.0, 0.0), "lin_vel_y": (y, y), "ang_vel_z": (yaw, yaw)})

    start = x_min
    eps = 1.0e-6
    while start < x_max - eps:
        end = min(start + width, x_max)
        bins.append({"lin_vel_x": (start, end), "lin_vel_y": (y, y), "ang_vel_z": (yaw, yaw)})
        start = end

    return bins


def curriculum_bins_from_params(params: CommandBinCurriculumParameters, *, device: torch.device | str) -> torch.Tensor:
    """Return explicit bin tensor, using configured bins only as an advanced override."""

    bins = params.bins if params.bins is not None else generated_bins_from_params(params)
    return bins_from_config(bins, device=device)


def bins_from_config(
    bins: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert bin mappings to a tensor shaped ``[num_bins, 6]``.

    Columns are ``x_min, x_max, y_min, y_max, yaw_min, yaw_max``.
    """

    if not bins:
        raise ValueError("Command curriculum requires at least one bin.")
    rows = []
    for index, bin_cfg in enumerate(bins):
        try:
            x_min, x_max = bin_cfg["lin_vel_x"]
            y_min, y_max = bin_cfg["lin_vel_y"]
            yaw_min, yaw_max = bin_cfg["ang_vel_z"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid command curriculum bin at index {index}: {bin_cfg!r}") from exc
        rows.append([float(x_min), float(x_max), float(y_min), float(y_max), float(yaw_min), float(yaw_max)])
    tensor = torch.tensor(rows, dtype=torch.float32, device=device)
    if (
        torch.any(tensor[:, 1] < tensor[:, 0])
        or torch.any(tensor[:, 3] < tensor[:, 2])
        or torch.any(tensor[:, 5] < tensor[:, 4])
    ):
        raise ValueError("Command curriculum bin max values must be >= min values.")
    return tensor


@torch.jit.script
def sample_binned_velocity_commands(
    bins: torch.Tensor,
    unlocked_bin: torch.Tensor,
    count: int,
    sample_only_frontier: bool,
    prefer_newer_bins: bool,
    older_bin_probability_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample velocity commands from unlocked bins."""

    max_bin = int(torch.clamp(unlocked_bin, min=0, max=bins.shape[0] - 1).item())
    if sample_only_frontier:
        bin_ids = torch.full((count,), max_bin, dtype=torch.long, device=bins.device)
    elif prefer_newer_bins and max_bin > 0:
        bin_index = torch.arange(max_bin + 1, dtype=torch.long, device=bins.device)
        distance_from_frontier = (max_bin - bin_index).to(dtype=torch.float32)
        decay = torch.tensor(older_bin_probability_decay, dtype=torch.float32, device=bins.device).clamp(0.01, 1.0)
        weights = torch.pow(decay, distance_from_frontier)
        cdf = torch.cumsum(weights / torch.sum(weights), dim=0)
        draws = torch.rand(count, dtype=torch.float32, device=bins.device)
        bin_ids = torch.searchsorted(cdf, draws).clamp(0, max_bin).to(dtype=torch.long)
    else:
        bin_ids = torch.randint(0, max_bin + 1, (count,), dtype=torch.long, device=bins.device)

    selected = bins[bin_ids]
    rand = torch.rand((count, 3), dtype=torch.float32, device=bins.device)
    commands = torch.empty((count, 3), dtype=torch.float32, device=bins.device)
    commands[:, 0] = selected[:, 0] + rand[:, 0] * (selected[:, 1] - selected[:, 0])
    commands[:, 1] = selected[:, 2] + rand[:, 1] * (selected[:, 3] - selected[:, 2])
    commands[:, 2] = selected[:, 4] + rand[:, 2] * (selected[:, 5] - selected[:, 4])
    return commands, bin_ids


@torch.jit.script
def command_tracking_success(
    command: torch.Tensor,
    displacement_xy: torch.Tensor,
    heading_delta: torch.Tensor,
    duration: torch.Tensor,
    terminated: torch.Tensor,
    min_survival_duration: float,
    velocity_tolerance: float,
    yaw_rate_tolerance: float,
) -> torch.Tensor:
    """Return success mask for completed command attempts.

    Command-bin progression only checks forward x velocity.
    """

    safe_duration = torch.clamp(duration, min=1.0e-6)
    achieved_xy = displacement_xy / safe_duration.unsqueeze(1)
    x_error = torch.abs(achieved_xy[:, 0] - command[:, 0])
    survived = duration >= min_survival_duration
    return (~terminated) & survived & (x_error <= velocity_tolerance)


@torch.jit.script
def update_command_bin_curriculum(
    attempts: torch.Tensor,
    successes: torch.Tensor,
    unlocked_bin: torch.Tensor,
    bin_ids: torch.Tensor,
    success_mask: torch.Tensor,
    successes_to_unlock: int,
    min_attempts_to_unlock: int,
    min_success_rate_to_unlock: float,
    max_attempts_to_track: int,
    reset_counts_on_unlock: bool,
) -> torch.Tensor:
    """Update bin attempts/successes and return the new unlocked bin index."""

    if bin_ids.numel() > 0:
        one = torch.ones_like(bin_ids, dtype=torch.float32)
        batch_attempts = torch.zeros_like(attempts)
        batch_successes = torch.zeros_like(successes)
        batch_attempts.scatter_add_(0, bin_ids, one)
        batch_successes.scatter_add_(0, bin_ids, success_mask.to(dtype=torch.float32))
        if max_attempts_to_track > 0:
            window = float(max_attempts_to_track)
            touched = batch_attempts > 0.0
            existing_budget = torch.clamp(window - batch_attempts, min=0.0)
            scale = torch.where(
                touched & (attempts > 0.0),
                torch.minimum(torch.ones_like(attempts), existing_budget / torch.clamp(attempts, min=1.0e-6)),
                torch.ones_like(attempts),
            )
            attempts.mul_(scale)
            successes.mul_(scale)
        attempts.add_(batch_attempts)
        successes.add_(batch_successes)

    current = int(torch.clamp(unlocked_bin, min=0, max=attempts.shape[0] - 1).item())
    current_attempts = float(attempts[current].item())
    current_successes = float(successes[current].item())
    success_rate = 0.0
    if current_attempts > 0:
        success_rate = current_successes / current_attempts
    can_unlock = (
        current_attempts >= float(min_attempts_to_unlock)
        and current_successes >= float(successes_to_unlock)
        and success_rate >= min_success_rate_to_unlock
        and current < attempts.shape[0] - 1
    )
    if can_unlock:
        current += 1
        if reset_counts_on_unlock:
            attempts.zero_()
            successes.zero_()
    unlocked_bin.fill_(current)
    return unlocked_bin


class CommandBinCurriculumState:
    """Small mutable state holder used outside ManagerBasedRLEnv, e.g. PSO."""

    def __init__(
        self,
        params: CommandBinCurriculumParameters,
        *,
        device: torch.device | str,
        prefer_newer_bins: bool = False,
        older_bin_probability_decay: float = 0.5,
    ):
        self.params = params
        self.prefer_newer_bins = bool(prefer_newer_bins)
        self.older_bin_probability_decay = float(older_bin_probability_decay)
        self.bins = curriculum_bins_from_params(params, device=device)
        initial = max(0, min(int(params.initial_unlocked_bin), self.bins.shape[0] - 1))
        self.unlocked_bin = torch.tensor(initial, dtype=torch.long, device=device)
        self.attempts = torch.zeros(self.bins.shape[0], dtype=torch.float32, device=device)
        self.successes = torch.zeros(self.bins.shape[0], dtype=torch.float32, device=device)

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_binned_velocity_commands(
            self.bins,
            self.unlocked_bin,
            int(count),
            bool(self.params.sample_only_frontier),
            self.prefer_newer_bins,
            self.older_bin_probability_decay,
        )

    def update(self, bin_ids: torch.Tensor, success_mask: torch.Tensor) -> None:
        update_command_bin_curriculum(
            self.attempts,
            self.successes,
            self.unlocked_bin,
            bin_ids.to(device=self.bins.device, dtype=torch.long),
            success_mask.to(device=self.bins.device, dtype=torch.bool),
            int(self.params.successes_to_unlock),
            int(self.params.min_attempts_to_unlock),
            float(self.params.min_success_rate_to_unlock),
            int(self.params.max_attempts_to_track),
            bool(self.params.reset_counts_on_unlock),
        )

    def summary(self) -> dict[str, Any]:
        current = int(self.unlocked_bin.detach().cpu())
        max_bin = max(1, int(self.bins.shape[0]) - 1)
        current_attempts = float(self.attempts[current].detach().cpu())
        current_successes = float(self.successes[current].detach().cpu())
        current_success_rate = current_successes / current_attempts if current_attempts > 0.0 else 0.0
        return {
            "unlocked_bin": current,
            "num_bins": int(self.bins.shape[0]),
            "progress_percent": float(current / max_bin * 100.0),
            "current_bin_attempts": current_attempts,
            "current_bin_successes": current_successes,
            "current_bin_success_rate": current_success_rate,
            "attempts": [float(value) for value in self.attempts.detach().cpu().tolist()],
            "successes": [float(value) for value in self.successes.detach().cpu().tolist()],
            "prefer_newer_bins": self.prefer_newer_bins,
            "older_bin_probability_decay": self.older_bin_probability_decay,
        }
