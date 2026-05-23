from __future__ import annotations

from typing import NamedTuple

import torch

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


class TendonCoordinates(NamedTuple):
    """Coordinate tensors shared by all tendon length calculations.

    NamedTuple keeps field access readable in eager debug code while remaining
    usable from TorchScript.
    """

    joint_angles: torch.Tensor
    joint_angles_signed: torch.Tensor
    thetas: torch.Tensor
    theta_hats: torch.Tensor
    qs: torch.Tensor
    qhats: torch.Tensor


@torch.jit.script
def compute_tendon_coordinates(joint_angles: torch.Tensor, tendon_data: TendonDataJIT) -> TendonCoordinates:
    """Transform Isaac joint angles into the theta/q coordinates used by the tendon model."""
    joint_angles_signed = tendon_data.joint_directions * joint_angles

    thetas = torch.empty_like(tendon_data.tendon_offsets_theta)
    theta_ids = [
        tids.I_THETA_GST_3,
        tids.I_THETA_GST_4,
        tids.I_THETA_GST_5,
        tids.I_THETA_ALL_6,
        tids.I_THETA_DFT_5,
        tids.I_THETA_EDT1_4,
        tids.I_THETA_EDT1_5,
        tids.I_THETA_EDT2_4,
        tids.I_THETA_EDT2_5,
        tids.I_THETA_KFT_3,
        tids.I_THETA_KFT_8,
    ]
    joint_ids = [
        tids.I_JOINT_3,
        tids.I_JOINT_4,
        tids.I_JOINT_5,
        tids.I_JOINT_6,
        tids.I_JOINT_5,
        tids.I_JOINT_4,
        tids.I_JOINT_5,
        tids.I_JOINT_4,
        tids.I_JOINT_5,
        tids.I_JOINT_3,
        tids.I_JOINT_8,
    ]
    thetas[:, theta_ids] = joint_angles_signed[:, joint_ids] + tendon_data.tendon_offsets_theta[:, theta_ids]

    qs = torch.empty_like(tendon_data.tendon_offsets_q_theta)
    q_ids = [
        tids.I_Q_GST_3,
        tids.I_Q_GST_4,
        tids.I_Q_GST_5,
        tids.I_Q_GST_6,
        tids.I_Q_DFT_5,
        tids.I_Q_DFT_6,
    ]
    theta_for_q_ids = [
        tids.I_THETA_GST_3,
        tids.I_THETA_GST_4,
        tids.I_THETA_GST_5,
        tids.I_THETA_ALL_6,
        tids.I_THETA_DFT_5,
        tids.I_THETA_ALL_6,
    ]
    qs[:, q_ids] = thetas[:, theta_for_q_ids] + tendon_data.tendon_offsets_q_theta[:, q_ids]

    theta_hats = -thetas + 2 * torch.pi

    qhats = torch.empty_like(tendon_data.tendon_offsets_qhat_thetahat)
    qhats[:, tids.I_QHAT_EDT2_6] = (
        theta_hats[:, tids.I_THETA_ALL_6]
        + tendon_data.tendon_offsets_qhat_thetahat[:, tids.I_QHAT_EDT2_6]
    )

    return TendonCoordinates(
        joint_angles=joint_angles,
        joint_angles_signed=joint_angles_signed,
        thetas=thetas,
        theta_hats=theta_hats,
        qs=qs,
        qhats=qhats,
    )
