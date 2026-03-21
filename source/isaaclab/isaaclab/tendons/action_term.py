"""Tendon-based action term implementation."""

from abc import abstractmethod
from typing import Sequence
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.tendons.action_term_cfg import TendonActionTermCfg
from isaaclab.tendons.constants import (
    N_LINKS_PER_LEG,
    TendonData,
    tids,
    JOINT_AXIS_IDX,
    link_names_left,
    link_names_right,
    joint_names_left,
    joint_names_right,
)
from isaaclab.tendons.gst_manager import GSTTendonManager
from isaaclab.utils import configclass
import torch


class TendonActionTerm(ActionTerm):
    """Tendon-based action term for controlling tendon actuators."""

    def __init__(self, cfg: TendonActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Additional initialization for tendon actions can be added here

        self.robot = env.scene.articulations[cfg.asset_name]

        self.gst_tendon_manager = GSTTendonManager(
            robot=self.robot,
            tendon_data=TendonData(
                batch_size=env.num_envs,
                randomization_ranges=cfg.randomization_ranges,
            ),
        )

        self._raw_actions: torch.Tensor = torch.zeros(
            (env.num_envs, self.action_dim), device=env.device
        )
        self._processed_actions: torch.Tensor = torch.zeros(
            (env.num_envs, self.action_dim), device=env.device
        )

        self.link_indices_left_right, _ = self.robot.find_bodies(
            link_names_left + link_names_right, preserve_order=True
        )
        self.joint_indices_left, _ = self.robot.find_joints(
            joint_names_left, preserve_order=True
        )
        self.joint_indices_right, _ = self.robot.find_joints(
            joint_names_right, preserve_order=True
        )

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
            gst_tendon_torques_left,
            gst_tendon_torques_right,
        ) = (  # pyright: ignore[reportAssignmentType]
            self.gst_tendon_manager.compute_torques_jit()
        )

        batch_size = self.robot.num_instances
        # 4) apply torques: with axis [0 -1 0], to each link
        tendon_torques_full = torch.zeros(
            (batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device
        )

        # Fix coordinate system inversion for joint 3 and 4
        gst_tendon_torques_left[:, tids.I_JOINT_3] *= -1.0
        gst_tendon_torques_right[:, tids.I_JOINT_3] *= -1.0
        gst_tendon_torques_left[:, tids.I_JOINT_4] *= -1.0
        gst_tendon_torques_right[:, tids.I_JOINT_4] *= -1.0

        # Apply GST torques
        tendon_torques_full[:, : N_LINKS_PER_LEG - 1, JOINT_AXIS_IDX] = (
            -gst_tendon_torques_left
        )
        tendon_torques_full[
            :, 1:N_LINKS_PER_LEG, JOINT_AXIS_IDX
        ] += gst_tendon_torques_left
        tendon_torques_full[
            :, N_LINKS_PER_LEG : N_LINKS_PER_LEG * 2 - 1, JOINT_AXIS_IDX
        ] = -gst_tendon_torques_right
        tendon_torques_full[
            :, N_LINKS_PER_LEG + 1 :, JOINT_AXIS_IDX
        ] += gst_tendon_torques_right

        # Apply knee motor torques from actions
        tendon_torques_full[
            :, self.link_indices_left_right[tids.I_LINK_23], JOINT_AXIS_IDX
        ] = -self._processed_actions[:, 0]
        tendon_torques_full[
            :, self.link_indices_left_right[tids.I_LINK_34], JOINT_AXIS_IDX
        ] += self._processed_actions[:, 0]
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_LINK_23 + N_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] = -self._processed_actions[:, 1]
        tendon_torques_full[
            :,
            self.link_indices_left_right[tids.I_LINK_34 + N_LINKS_PER_LEG],
            JOINT_AXIS_IDX,
        ] += self._processed_actions[:, 1]

        self.robot.set_external_force_and_torque(
            forces=torch.zeros(
                (batch_size, N_LINKS_PER_LEG * 2, 3), device=self.device
            ),
            torques=tendon_torques_full,
            body_ids=self.link_indices_left_right,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the manager term.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # self.gst_tendon_manager.tendon_data.reset(env_ids)
