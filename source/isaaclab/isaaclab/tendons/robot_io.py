from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse

from isaaclab.tendons.models.analytic.constants import (
    N_CHAIN_LINKS_PER_LEG,
    joint_names_left,
    joint_names_right,
    link_names_left,
    link_names_right,
)
import isaaclab.tendons.models.analytic.indices as tids


class TendonRobotIO:
    """Isaac-specific body/joint lookup and external-force construction."""

    def __init__(self, robot):
        self.robot = robot
        self.device = robot.device

        self.link_indices_left_right, _ = robot.find_bodies(link_names_left + link_names_right, preserve_order=True)
        self.joint_indices_left, _ = robot.find_joints(joint_names_left, preserve_order=True)
        self.joint_indices_right, _ = robot.find_joints(joint_names_right, preserve_order=True)

        self.hip_joint_names = [
            "l2_pseudo_acetabulofemoral_flexion",
            "r2_pseudo_acetabulofemoral_flexion",
        ]
        self.hip_static_joint_names = [
            "r0_acetabulofemoral_roll",
            "r1_acetabulofemoral_lateral",
            "l0_acetabulofemoral_roll",
            "l1_acetabulofemoral_lateral",
        ]
        self.hip_joint_indices, _ = robot.find_joints(self.hip_joint_names, preserve_order=True)
        self.hip_static_joint_indices, _ = robot.find_joints(self.hip_static_joint_names, preserve_order=True)

        self.foot_link_names = [
            link_names_left[tids.I_CHAIN_LINK_67],
            link_names_right[tids.I_CHAIN_LINK_67],
        ]
        self.foot_link_indices, _ = robot.find_bodies(self.foot_link_names, preserve_order=True)

    def get_leg_joint_angles(self, *, requires_grad: bool = True) -> torch.Tensor:
        joint_angles = torch.cat(
            (
                self.robot.data.joint_pos[:, self.joint_indices_left].clone(),
                self.robot.data.joint_pos[:, self.joint_indices_right].clone(),
            ),
            dim=0,
        )
        if requires_grad:
            joint_angles.requires_grad_(True)
        return joint_angles

    def empty_external_forces(self) -> torch.Tensor:
        return torch.zeros((self.robot.num_instances, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device)

    def compute_external_forces(self, *, virtual_ground_height: float | None = None) -> torch.Tensor:
        forces = self.empty_external_forces()
        if virtual_ground_height is None:
            return forces

        foot_heights = self.robot.data.body_com_pos_w[:, self.foot_link_indices, 2]
        penetration_depths = virtual_ground_height - foot_heights
        weight = 400.0
        forces_world = penetration_depths.clamp(min=0.0).unsqueeze(-1) * torch.tensor(
            [0.0, 0.0, weight * 20.0],
            device=self.device,
        )
        forces[:, [tids.I_CHAIN_LINK_67, tids.I_CHAIN_LINK_67 + N_CHAIN_LINKS_PER_LEG], :] = quat_apply_inverse(
            self.robot.data.body_link_quat_w[:, self.foot_link_indices],
            forces_world,
        )
        return forces
