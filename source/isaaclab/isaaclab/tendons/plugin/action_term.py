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

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.tendons.manager import TendonManager
from isaaclab.tendons.models.analytic.constants import N_CHAIN_LINKS_PER_LEG  # NOTE: Changed from N_LINKS_PER_LEG
from isaaclab.tendons.models.analytic.constants import (
    JOINT_AXIS_IDX,
    hip_joint_names,
    joint_names_left,
    joint_names_right,
    link_names_left,
    link_names_right,
    tids,
)
from isaaclab.tendons.models.analytic.tendon_data import TendonData

if TYPE_CHECKING:
    from isaaclab.tendons.plugin.action_term_cfg import TendonActionTermCfg, TendonActionTermHybridCfg


class TendonActionTermHybrid(ActionTerm):
    """Passive tendon dynamics action term.

    Important:
        This term intentionally exposes zero policy actions. It is registered as
        an ActionTerm only so Isaac Lab calls ``apply_actions()`` every physics
        step. The tendon torques are passive dynamics computed from the current
        joint state.

    This mirrors the JIT path used in ``run.py``:

        tendon_manager.apply_jit(virtual_ground_height=None, dt=SIM_DT)
    """

    def __init__(self, cfg: TendonActionTermHybridCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot = env.scene.articulations[cfg.asset_name]
        self._env = env

        self.tendon_manager = TendonManager(
            robot=self.robot,
            tendon_data=TendonData(
                batch_size=env.num_envs,
                randomization_ranges=cfg.randomization_ranges,
            ),
        )

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
        self.tendon_manager.apply_jit(
            virtual_ground_height=None,
            dt=self._get_physics_dt(),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset tendon internal state.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all
                environments are considered.
        """
        # The manager stores previous tendon delta lengths for damping. Clear it
        # on reset so new episodes do not inherit damping state from old ones.
        self.tendon_manager._prev_delta_lengths = None

        # Keep this guarded because older TendonData versions may not expose reset.
        if hasattr(self.tendon_manager.tendon_data, "reset"):
            self.tendon_manager.tendon_data.reset(env_ids)

    def _get_physics_dt(self) -> float:
        """Best-effort physics dt lookup for manager-based envs.

        In the standalone script this value is passed as SIM_DT. In Isaac Lab
        environments the exact attribute name can vary, so this helper tries the
        common locations and falls back to 0.0 if none are available.
        """
        for attr_name in ("physics_dt", "step_dt", "dt"):
            value = getattr(self._env, attr_name, None)
            if value is None:
                continue
            if callable(value):
                value = value()
            if value is not None:
                return float(value)

        sim = getattr(self._env, "sim", None)
        if sim is not None and hasattr(sim, "get_physics_dt"):
            return float(sim.get_physics_dt())

        # Fallback: no damping contribution in TendonManager.
        return 0.0


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

        self.tendon_manager = TendonManager(
            robot=self.robot,
            tendon_data=TendonData(
                batch_size=env.num_envs,
                randomization_ranges=cfg.randomization_ranges,
            ),
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
        (
            tendon_torques_left,
            tendon_torques_right,
        ) = self.tendon_manager.compute_torques_jit()  # pyright: ignore[reportAssignmentType]

        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros((batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device)

        # Fix coordinate system inversion for joint 3 and 4
        tendon_torques_left[:, tids.I_Q_GST_3] *= -1.0
        tendon_torques_right[:, tids.I_Q_GST_3] *= -1.0
        tendon_torques_left[:, tids.I_Q_GST_4] *= -1.0
        tendon_torques_right[:, tids.I_Q_GST_4] *= -1.0

        # Apply GST torques
        tendon_torques_full[:, : N_CHAIN_LINKS_PER_LEG - 1, JOINT_AXIS_IDX] = -tendon_torques_left
        tendon_torques_full[:, 1:N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] += tendon_torques_left
        tendon_torques_full[:, N_CHAIN_LINKS_PER_LEG : N_CHAIN_LINKS_PER_LEG * 2 - 1, JOINT_AXIS_IDX] = (
            -tendon_torques_right
        )
        tendon_torques_full[:, N_CHAIN_LINKS_PER_LEG + 1 :, JOINT_AXIS_IDX] += tendon_torques_right

        # Apply knee motor torques from actions
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_CONNECTOR_LINK_GST_23],
            JOINT_AXIS_IDX,
        ] = -self._processed_actions[:, 0]
        tendon_torques_full[:, self.link_indices_left_right[tids.I_LINK_34], JOINT_AXIS_IDX] += self._processed_actions[
            :, 0
        ]
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_CONNECTOR_LINK_GST_23 + N_CHAIN_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] = -self._processed_actions[:, 1]
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_LINK_34 + N_CHAIN_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] += self._processed_actions[:, 1]

        self.robot.set_external_force_and_torque(
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
        self.tendon_manager._prev_delta_lengths = None
        if hasattr(self.tendon_manager.tendon_data, "reset"):
            self.tendon_manager.tendon_data.reset(env_ids)
