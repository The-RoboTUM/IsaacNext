from __future__ import annotations

import torch

from isaaclab.tendons.models.analytic.constants import JOINT_AXIS_IDX, N_CHAIN_LINKS_PER_LEG
import isaaclab.tendons.models.analytic.indices as tids


class TendonTorqueMapper:
    """Map per-joint tendon torques to external torques on Isaac links."""

    def __init__(self, device):
        self.device = device

    def joint_to_link_torques(
        self,
        tendon_torques_joints_left: torch.Tensor,
        tendon_torques_joints_right: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        left = tendon_torques_joints_left.clone()
        right = tendon_torques_joints_right.clone()

        # Preserve the direction convention from the old apply_debug implementation.
        left[:, tids.I_JOINT_3] *= -1.0
        right[:, tids.I_JOINT_3] *= -1.0
        left[:, tids.I_JOINT_4] *= -1.0
        right[:, tids.I_JOINT_4] *= -1.0

        link_torques = torch.zeros((batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3), device=self.device)

        left_parent_links = (
            tids.I_CHAIN_LINK_23,
            tids.I_CHAIN_LINK_34,
            tids.I_CHAIN_LINK_4prime5,
            tids.I_CHAIN_LINK_56,
        )
        right_parent_links = tuple(i + N_CHAIN_LINKS_PER_LEG for i in left_parent_links)
        flexion_joint_ids = (tids.I_JOINT_3, tids.I_JOINT_4, tids.I_JOINT_5, tids.I_JOINT_6)

        link_torques[:, left_parent_links, JOINT_AXIS_IDX] = left[:, flexion_joint_ids]
        link_torques[:, right_parent_links, JOINT_AXIS_IDX] = right[:, flexion_joint_ids]

        # KFT joint 8 acts on link 23 as parent and link 38 as child.
        link_torques[:, (tids.I_CHAIN_LINK_23,), JOINT_AXIS_IDX] += left[:, (tids.I_JOINT_8,)]
        link_torques[:, (tids.I_CHAIN_LINK_23 + N_CHAIN_LINKS_PER_LEG,), JOINT_AXIS_IDX] += right[:, (tids.I_JOINT_8,)]

        left_child_links = (
            tids.I_CHAIN_LINK_34,
            tids.I_CHAIN_LINK_4prime5,
            tids.I_CHAIN_LINK_56,
            tids.I_CHAIN_LINK_67,
            tids.I_CHAIN_LINK_38,
        )
        right_child_links = tuple(i + N_CHAIN_LINKS_PER_LEG for i in left_child_links)
        child_joint_ids = (tids.I_JOINT_3, tids.I_JOINT_4, tids.I_JOINT_5, tids.I_JOINT_6, tids.I_JOINT_8)

        link_torques[:, left_child_links, JOINT_AXIS_IDX] -= left[:, child_joint_ids]
        link_torques[:, right_child_links, JOINT_AXIS_IDX] -= right[:, child_joint_ids]

        return link_torques
