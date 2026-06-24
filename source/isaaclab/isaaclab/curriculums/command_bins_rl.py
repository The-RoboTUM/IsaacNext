# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL command-generator wrapper for the shared binned command curriculum."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.curriculums.command_bins import (
    CommandBinCurriculumParameters,
    CommandBinCurriculumState,
    command_tracking_success,
)
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BinnedVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command generator backed by shared command-bin curriculum state."""

    cfg: Any

    def __init__(self, cfg: Any, env):
        super().__init__(cfg, env)
        self._curriculum_state = CommandBinCurriculumState(cfg.curriculum, device=self.device)
        self.bin_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_start_pos_w = self.robot.data.root_pos_w[:, :2].clone()
        self.command_start_heading_w = self.robot.data.heading_w.clone()
        self.command_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["curriculum_unlocked_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_progress_percent"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_current_bin_attempts"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_current_bin_successes"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_current_bin_success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampled_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampled_bin_percent"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["command_duration_s"] = torch.zeros(self.num_envs, device=self.device)
        self._update_curriculum_metrics()

    def _bin_percent(self, bin_ids: torch.Tensor) -> torch.Tensor:
        max_bin = max(1, int(self._curriculum_state.bins.shape[0]) - 1)
        return bin_ids.to(dtype=torch.float32) / float(max_bin) * 100.0

    def _update_curriculum_metrics(self) -> None:
        unlocked_index = int(self._curriculum_state.unlocked_bin.detach().cpu())
        unlocked = self._curriculum_state.unlocked_bin.to(dtype=torch.float32)
        current_attempts = self._curriculum_state.attempts[unlocked_index]
        current_successes = self._curriculum_state.successes[unlocked_index]
        current_success_rate = torch.where(
            current_attempts > 0.0,
            current_successes / current_attempts,
            torch.zeros_like(current_attempts),
        )
        self.metrics["curriculum_unlocked_bin"].fill_(float(unlocked.detach().cpu()))
        self.metrics["curriculum_progress_percent"].fill_(
            float(self._bin_percent(self._curriculum_state.unlocked_bin).detach().cpu())
        )
        self.metrics["curriculum_current_bin_attempts"].fill_(float(current_attempts.detach().cpu()))
        self.metrics["curriculum_current_bin_successes"].fill_(float(current_successes.detach().cpu()))
        self.metrics["curriculum_current_bin_success_rate"].fill_(float(current_success_rate.detach().cpu()))
        self.metrics["sampled_bin"][:] = self.bin_ids.to(dtype=torch.float32)
        self.metrics["sampled_bin_percent"][:] = self._bin_percent(self.bin_ids)

    def _record_attempts(self, env_ids: torch.Tensor, *, terminated: torch.Tensor | None = None) -> None:
        if env_ids.numel() == 0:
            return
        valid = (self.command_counter[env_ids] > 0) & (
            self._env.episode_length_buf[env_ids] > self.command_start_step[env_ids]
        )
        env_ids = env_ids[valid]
        if env_ids.numel() == 0:
            return

        displacement_xy = self.robot.data.root_pos_w[env_ids, :2] - self.command_start_pos_w[env_ids]
        heading_delta = wrap_to_pi(self.robot.data.heading_w[env_ids] - self.command_start_heading_w[env_ids])
        command_steps = torch.clamp(self._env.episode_length_buf[env_ids] - self.command_start_step[env_ids], min=1)
        duration = command_steps.to(dtype=torch.float32) * float(self._env.step_dt)
        self.metrics["command_duration_s"][env_ids] = duration
        if terminated is None:
            terminated = torch.zeros(env_ids.shape, dtype=torch.bool, device=self.device)
        else:
            terminated = terminated.to(device=self.device, dtype=torch.bool)
        max_command_duration = float(self.cfg.resampling_time_range[1])
        min_survival_duration = float(self.cfg.curriculum.success_min_survival_fraction) * max_command_duration
        success_mask = command_tracking_success(
            self.vel_command_b[env_ids],
            displacement_xy,
            heading_delta,
            duration,
            terminated,
            min_survival_duration,
            float(self.cfg.curriculum.success_velocity_tolerance),
            float(self.cfg.curriculum.success_yaw_rate_tolerance),
        )
        self._curriculum_state.update(self.bin_ids[env_ids], success_mask)
        self.command_start_step[env_ids] = self._env.episode_length_buf[env_ids]
        self._update_curriculum_metrics()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset command metrics and sample a new command for reset environments.

        ManagerBasedRLEnv clears ``episode_length_buf`` after manager resets, so
        the base reset path samples commands while the old episode length is
        still present. Override the command-start step for reset commands so the
        next curriculum update measures duration from the new episode start.
        """

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_tensor.numel() > 0 and hasattr(self._env, "reset_terminated"):
            self._record_attempts(env_ids_tensor, terminated=self._env.reset_terminated[env_ids_tensor])

        extras = super().reset(env_ids)
        if env_ids is None:
            self.command_start_step[:] = 0
        else:
            if env_ids_tensor.numel() > 0:
                self.command_start_step[env_ids_tensor] = 0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_tensor.numel() == 0:
            return
        self._record_attempts(env_ids_tensor)
        commands, bin_ids = self._curriculum_state.sample(int(env_ids_tensor.numel()))
        self.vel_command_b[env_ids_tensor] = commands
        self.bin_ids[env_ids_tensor] = bin_ids
        self.command_start_pos_w[env_ids_tensor] = self.robot.data.root_pos_w[env_ids_tensor, :2]
        self.command_start_heading_w[env_ids_tensor] = self.robot.data.heading_w[env_ids_tensor]
        self.command_start_step[env_ids_tensor] = self._env.episode_length_buf[env_ids_tensor]
        self.is_standing_env[env_ids_tensor] = False
        self.is_heading_env[env_ids_tensor] = False
        self._update_curriculum_metrics()

    def update_curriculum(self, env, env_ids: Sequence[int]) -> dict[str, Any]:
        return self._curriculum_state.summary()


@configclass
class BinnedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command sampled from progressively unlocked bins."""

    class_type: type = BinnedVelocityCommand
    curriculum: CommandBinCurriculumParameters = field(default_factory=CommandBinCurriculumParameters)


def update_binned_velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
) -> dict[str, Any]:
    """Isaac Lab curriculum term for the shared binned velocity command."""

    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "update_curriculum"):
        return {}
    return command_term.update_curriculum(env, env_ids)


def make_binned_velocity_curriculum_term(command_name: str = "base_velocity") -> CurrTerm:
    return CurrTerm(
        func=update_binned_velocity_command_curriculum,
        params={"command_name": command_name},
    )
