import torch

from isaaclab.tendons.models.analytic.constants import JOINT_AXIS_IDX, N_CHAIN_LINKS_PER_LEG
import isaaclab.tendons.models.analytic.indices as tids


@torch.jit.script
def joint_to_link_torques_jit_kernel(
    tendon_torques_joints_left: torch.Tensor,
    tendon_torques_joints_right: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """TorchScript tensor-only joint-to-link torque mapping."""
    N_CHAIN_LINKS_PER_LEG: int = 6
    JOINT_AXIS_IDX: int = 0
    I_CHAIN_LINK_23: int = 0
    I_CHAIN_LINK_34: int = 1
    I_CHAIN_LINK_38: int = 5
    I_CHAIN_LINK_4prime5: int = 2
    I_CHAIN_LINK_56: int = 3
    I_CHAIN_LINK_67: int = 4
    I_JOINT_3: int = 0
    I_JOINT_4: int = 1
    I_JOINT_5: int = 2
    I_JOINT_6: int = 3
    I_JOINT_8: int = 4
    left = tendon_torques_joints_left.clone()
    right = tendon_torques_joints_right.clone()

    # Preserve the direction convention from the old apply_debug implementation.
    left[:, I_JOINT_3] = left[:, I_JOINT_3] * -1.0
    right[:, I_JOINT_3] = right[:, I_JOINT_3] * -1.0
    left[:, I_JOINT_4] = left[:, I_JOINT_4] * -1.0
    right[:, I_JOINT_4] = right[:, I_JOINT_4] * -1.0

    link_torques = left.new_zeros((batch_size, N_CHAIN_LINKS_PER_LEG * 2, 3))

    link_torques[:, I_CHAIN_LINK_23, JOINT_AXIS_IDX] = left[:, I_JOINT_3]
    link_torques[:, I_CHAIN_LINK_34, JOINT_AXIS_IDX] = left[:, I_JOINT_4]
    link_torques[:, I_CHAIN_LINK_4prime5, JOINT_AXIS_IDX] = left[:, I_JOINT_5]
    link_torques[:, I_CHAIN_LINK_56, JOINT_AXIS_IDX] = left[:, I_JOINT_6]

    link_torques[:, I_CHAIN_LINK_23 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] = right[:, I_JOINT_3]
    link_torques[:, I_CHAIN_LINK_34 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] = right[:, I_JOINT_4]
    link_torques[:, I_CHAIN_LINK_4prime5 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] = right[:, I_JOINT_5]
    link_torques[:, I_CHAIN_LINK_56 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] = right[:, I_JOINT_6]

    # KFT joint 8 acts on link 23 as parent and link 38 as child.
    link_torques[:, I_CHAIN_LINK_23, JOINT_AXIS_IDX] += left[:, I_JOINT_8]
    link_torques[:, I_CHAIN_LINK_23 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] += right[:, I_JOINT_8]

    link_torques[:, I_CHAIN_LINK_34, JOINT_AXIS_IDX] -= left[:, I_JOINT_3]
    link_torques[:, I_CHAIN_LINK_4prime5, JOINT_AXIS_IDX] -= left[:, I_JOINT_4]
    link_torques[:, I_CHAIN_LINK_56, JOINT_AXIS_IDX] -= left[:, I_JOINT_5]
    link_torques[:, I_CHAIN_LINK_67, JOINT_AXIS_IDX] -= left[:, I_JOINT_6]
    link_torques[:, I_CHAIN_LINK_38, JOINT_AXIS_IDX] -= left[:, I_JOINT_8]

    link_torques[:, I_CHAIN_LINK_34 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] -= right[:, I_JOINT_3]
    link_torques[:, I_CHAIN_LINK_4prime5 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] -= right[:, I_JOINT_4]
    link_torques[:, I_CHAIN_LINK_56 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] -= right[:, I_JOINT_5]
    link_torques[:, I_CHAIN_LINK_67 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] -= right[:, I_JOINT_6]
    link_torques[:, I_CHAIN_LINK_38 + N_CHAIN_LINKS_PER_LEG, JOINT_AXIS_IDX] -= right[:, I_JOINT_8]

    return link_torques


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

    def joint_to_link_torques_jit(
        self,
        tendon_torques_joints_left: torch.Tensor,
        tendon_torques_joints_right: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        return joint_to_link_torques_jit_kernel(
            tendon_torques_joints_left,
            tendon_torques_joints_right,
            batch_size,
        )
