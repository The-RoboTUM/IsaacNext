# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Tendon-based action term implementation.

This file keeps two action terms:

- TendonActionTermHybrid: passive tendon dynamics only. It has action_dim = 0 and
  applies the same JIT path used in run.py: TendonManager.apply_jit(...).
- TendonActionTerm: older policy-controlled tendon torque action term. Kept for
  compatibility/debugging, but the passive hybrid term is the one to use when the
  policy should not command tendon actions.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.tendons.manager import TendonManager
from isaaclab.tendons.models.analytic.constants import (
    JOINT_AXIS_IDX,
    N_CHAIN_LINKS_PER_LEG,  # NOTE: Changed from N_LINKS_PER_LEG
    hip_joint_names,
    joint_names_left,
    joint_names_right,
    link_names_left,
    link_names_right,
    tids,
)
from isaaclab.tendons.models.analytic.tendon_data import TendonData
from isaaclab.tendons.models.identix import IdentixForrestElasticTendonModel
from isaaclab.tendons.parameter_loader import load_forrest_parameter_config

if TYPE_CHECKING:
    from isaaclab.tendons.plugin.action_term_cfg import TendonActionTermCfg, TendonActionTermHybridCfg


def _get_physics_dt(env: ManagerBasedEnv) -> float:
    """Best-effort physics dt lookup for manager-based envs."""
    for attr_name in ("physics_dt", "step_dt", "dt"):
        value = getattr(env, attr_name, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if value is not None:
            return float(value)

    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim, "get_physics_dt"):
        return float(sim.get_physics_dt())

    # Fallback: no damping contribution in TendonManager.
    return 0.0


class TendonActionTermHybrid(ActionTerm):
    """Passive tendon dynamics action term.

    Important:
        This term intentionally exposes zero policy actions. It is registered as
        an ActionTerm only so Isaac Lab calls ``apply_actions()`` every physics
        step. The tendon torques are passive dynamics computed from the current
        joint state.

    This mirrors the JIT path used in ``run.py``:

        tendon_manager.apply_jit(dt=SIM_DT)
    """

    def __init__(self, cfg: TendonActionTermHybridCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot = env.scene.articulations[cfg.asset_name]
        self._env = env
        forrest_params = load_forrest_parameter_config(cfg.parameters_file)
        tendon_constants = forrest_params.to_tendon_constants(device=env.device)
        action_params = forrest_params.training.actions
        model_type = str(cfg.model_type or action_params.tendon_model_type)
        model = None
        apply_mode = "link_wrench"
        if model_type == "identix_elastic":
            apply_mode = (
                action_params.tendon_identix_apply_mode
                if cfg.identix_apply_mode is None
                else str(cfg.identix_apply_mode)
            )
            model = IdentixForrestElasticTendonModel(
                self.robot,
                bundle_dir=cfg.identix_bundle_dir or action_params.tendon_identix_bundle_dir,
                identix_repo_path=cfg.identix_repo_path or action_params.tendon_identix_repo_path,
                compile=action_params.tendon_identix_compile
                if cfg.identix_compile is None
                else bool(cfg.identix_compile),
                transfer=action_params.tendon_identix_transfer
                if cfg.identix_transfer is None
                else cfg.identix_transfer,
                force_scale=action_params.tendon_identix_force_scale
                if cfg.identix_force_scale is None
                else float(cfg.identix_force_scale),
                force_sign=action_params.tendon_identix_force_sign
                if cfg.identix_force_sign is None
                else float(cfg.identix_force_sign),
            )
        elif model_type != "analytic":
            raise ValueError(f"Unsupported Forrest tendon model_type: {model_type!r}")

        self.tendon_manager = TendonManager(
            robot=self.robot,
            tendon_data=TendonData(
                batch_size=env.num_envs,
                randomization_ranges=cfg.randomization_ranges,
                tc=tendon_constants,
                device=env.device,
            ),
            model=model,
            tendon_damping=forrest_params.tendon_damping(),
            apply_mode=apply_mode,
        )
        self._update_interval = max(1, int(cfg.update_interval))
        self._apply_count = 0

        # Keep these tensors so the ActionManager/debug interface can query them.
        # Shape is (num_envs, 0) because this is passive-only.
        self._raw_actions = torch.zeros((env.num_envs, self.action_dim), device=env.device)
        self._processed_actions = torch.zeros((env.num_envs, self.action_dim), device=env.device)

        # Kept for debugging / compatibility with older code that inspected these.
        self.link_indices_left_right, _ = self.robot.find_bodies(
            link_names_left + link_names_right, preserve_order=True
        )
        self.joint_indices_left, _ = self.robot.find_joints(joint_names_left, preserve_order=True)
        self.joint_indices_right, _ = self.robot.find_joints(joint_names_right, preserve_order=True)
        self.hip_joint_indices, _ = self.robot.find_joints(hip_joint_names, preserve_order=True)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term.

        Zero is intentional: the policy does not command the tendons.
        """
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """No-op because this term is passive-only.

        Isaac Lab may pass an empty tensor with shape ``(num_envs, 0)``. We keep
        the method explicit so it is clear that policy actions are intentionally
        ignored here.
        """
        if actions.numel() != 0:
            raise ValueError(
                f"TendonActionTermHybrid is passive-only and expects zero actions, got shape {tuple(actions.shape)}."
            )

    def apply_actions(self):
        """Apply passive tendon dynamics using the same JIT path as run.py.

        Note:
            This is called at every simulation step by the Isaac Lab action
            manager. The actual torque mapping and external torque application
            are delegated to ``TendonManager.apply_jit`` so the env integration
            matches the standalone JIT simulation path.
        """
        profiler = getattr(self._env, "_external_profiler", None)
        self.tendon_manager._profile_external_profiler = profiler
        should_recompute = self._apply_count % self._update_interval == 0
        self._apply_count += 1
        if not should_recompute and self.tendon_manager.reapply_cached_jit():
            return

        if bool(getattr(profiler, "enabled", False)):
            with profiler.scope("env/tendon/apply_jit"):
                self.tendon_manager.apply_jit(
                    dt=self._get_physics_dt(),
                )
            return

        self.tendon_manager.apply_jit(dt=self._get_physics_dt())

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset tendon internal state.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all
                environments are considered.
        """
        # Keep this guarded because older TendonData versions may not expose reset.
        if hasattr(self.tendon_manager.tendon_data, "reset"):
            self.tendon_manager.tendon_data.reset(env_ids)
        self.tendon_manager.reset_damping_state(env_ids)
        self._apply_count = 0

    def _get_physics_dt(self) -> float:
        """Best-effort physics dt lookup for manager-based envs.

        In the standalone script this value is passed as SIM_DT. In Isaac Lab
        environments the exact attribute name can vary, so this helper tries the
        common locations and falls back to 0.0 if none are available.
        """
        return _get_physics_dt(self._env)


class TendonActionTerm(ActionTerm):
    """Tendon-based action term for controlling tendon actuators.

    This older term is policy-controlled and exposes two actions. It is kept for
    compatibility/debugging. For passive dynamics in the RL env, prefer
    ``TendonActionTermHybrid`` above.
    """

    def __init__(self, cfg: TendonActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Additional initialization for tendon actions can be added here

        self.robot = env.scene.articulations[cfg.asset_name]
        self._env = env
        forrest_params = load_forrest_parameter_config(cfg.parameters_file)
        tendon_constants = forrest_params.to_tendon_constants(device=env.device)

        self.tendon_manager = TendonManager(
            robot=self.robot,
            tendon_data=TendonData(
                batch_size=env.num_envs,
                randomization_ranges=cfg.randomization_ranges,
                tc=tendon_constants,
                device=env.device,
            ),
            tendon_damping=forrest_params.tendon_damping(),
        )

        self._raw_actions: torch.Tensor = torch.zeros((env.num_envs, self.action_dim), device=env.device)
        self._processed_actions: torch.Tensor = torch.zeros((env.num_envs, self.action_dim), device=env.device)

        self.link_indices_left_right, _ = self.robot.find_bodies(
            link_names_left + link_names_right, preserve_order=True
        )
        self.joint_indices_left, _ = self.robot.find_joints(joint_names_left, preserve_order=True)
        self.joint_indices_right, _ = self.robot.find_joints(joint_names_right, preserve_order=True)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions: (B, 2) for left and right knees
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self._raw_actions[:].clamp(max=0.0)

    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        profiler = getattr(self._env, "_external_profiler", None)
        self.tendon_manager._profile_external_profiler = profiler
        if bool(getattr(profiler, "enabled", False)):
            with profiler.scope("env/tendon/policy_controlled_apply"):
                self._apply_policy_controlled_tendon_actions()
            return

        self._apply_policy_controlled_tendon_actions()

    def _apply_policy_controlled_tendon_actions(self) -> None:
        (
            tendon_torques_left,
            tendon_torques_right,
        ) = self.tendon_manager.compute_torques_jit(dt=_get_physics_dt(self._env))  # pyright: ignore[reportAssignmentType]

        batch_size = self.robot.num_instances
        tendon_torques_full = self.tendon_manager.torque_mapper.joint_to_link_torques_jit(
            tendon_torques_left,
            tendon_torques_right,
            batch_size=batch_size,
        ).clone()

        # Apply knee motor torques from actions
        tendon_torques_full[:, tids.I_CHAIN_LINK_23, JOINT_AXIS_IDX] += -self._processed_actions[:, 0]
        tendon_torques_full[:, tids.I_CHAIN_LINK_34, JOINT_AXIS_IDX] += self._processed_actions[:, 0]
        tendon_torques_full[
            :,
            tids.I_CHAIN_LINK_23 + N_CHAIN_LINKS_PER_LEG,
            JOINT_AXIS_IDX,
        ] += -self._processed_actions[:, 1]
        tendon_torques_full[
            :,
            tids.I_CHAIN_LINK_34 + N_CHAIN_LINKS_PER_LEG,
            JOINT_AXIS_IDX,
        ] += self._processed_actions[:, 1]

        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=torch.zeros((batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device),
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if hasattr(self.tendon_manager.tendon_data, "reset"):
            self.tendon_manager.tendon_data.reset(env_ids)
        self.tendon_manager.reset_damping_state(env_ids)
