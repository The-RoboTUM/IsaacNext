# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import NamedTuple

import torch

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.geometry.common import TendonLengthOutput
from isaaclab.tendons.models.analytic.geometry.kinematics import TendonCoordinates
from isaaclab.tendons.models.analytic.geometry.shared import SharedTendonGeometry
from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


class DFTDeltaCoreOutput(NamedTuple):
    delta_l: torch.Tensor
    DFT_L_s: torch.Tensor
    DFT_state_A: torch.Tensor
    DFT_state_B: torch.Tensor
    DFT_state_C: torch.Tensor
    DFT_state_D: torch.Tensor
    DFT_L_s_A: torch.Tensor
    DFT_L_s_B: torch.Tensor
    DFT_L_s_C: torch.Tensor
    DFT_L_s_D: torch.Tensor
    DFT_q5_D: torch.Tensor
    DFT_q6_B: torch.Tensor


@torch.jit.script
def compute_dft_delta_l_core(
    coords: TendonCoordinates,
    geom: SharedTendonGeometry,
    tendon_data: TendonDataJIT,
) -> DFTDeltaCoreOutput:
    """DFT spring-length delta shared by debug and JIT paths."""
    thetas = coords.thetas
    qs = coords.qs

    DFT_h5_B_disengaged = geom.DFT_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5]
    DFT_h5_C_disengaged = geom.DFT_h5_C > tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5]
    DFT_h6_C_disengaged = geom.DFT_h6_C > tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6]
    DFT_h6_D_disengaged = geom.DFT_h6_D > tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6]

    DFT_state_C = (DFT_h5_B_disengaged & DFT_h6_C_disengaged) | (DFT_h6_D_disengaged & DFT_h5_C_disengaged)
    DFT_state_B = ~DFT_state_C & DFT_h5_B_disengaged
    DFT_state_D = ~DFT_state_C & DFT_h6_D_disengaged
    DFT_state_A = ~(DFT_state_B | DFT_state_C | DFT_state_D)

    DFT_L_s_A = (
        tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_C5]
        + qs[:, tids.I_Q_DFT_5] * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_56]
        + qs[:, tids.I_Q_DFT_6] * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_6C]
    )

    DFT_q6_B = (
        thetas[:, tids.I_THETA_ALL_6]
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_DFT_6C_J6]
        - 2 * torch.pi
        + geom.DFT_phi_4_B
        + thetas[:, tids.I_THETA_DFT_5]
    )
    DFT_L_s_B = (
        geom.DFT_l_c6
        + DFT_q6_B * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_6C]
    )

    DFT_L_s_C = torch.sqrt(geom.DFT_l_c7_squared)

    DFT_q5_D = (
        thetas[:, tids.I_THETA_DFT_5]
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_DFT_C5_J5]
        - geom.DFT_phi_5_D
    )
    DFT_L_s_D = (
        tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_C5]
        + DFT_q5_D * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5]
        + geom.DFT_l_57
    )

    DFT_L_s = torch.where(
        DFT_state_A,
        DFT_L_s_A,
        torch.where(
            DFT_state_B,
            DFT_L_s_B,
            torch.where(
                DFT_state_C,
                DFT_L_s_C,
                DFT_L_s_D,
            ),
        ),
    )

    DFT_delta_L_s = tendon_data.dft_length - DFT_L_s

    return DFTDeltaCoreOutput(
        delta_l=DFT_delta_L_s,
        DFT_L_s=DFT_L_s,
        DFT_state_A=DFT_state_A,
        DFT_state_B=DFT_state_B,
        DFT_state_C=DFT_state_C,
        DFT_state_D=DFT_state_D,
        DFT_L_s_A=DFT_L_s_A,
        DFT_L_s_B=DFT_L_s_B,
        DFT_L_s_C=DFT_L_s_C,
        DFT_L_s_D=DFT_L_s_D,
        DFT_q5_D=DFT_q5_D,
        DFT_q6_B=DFT_q6_B,
    )


def compute_dft_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    core = compute_dft_delta_l_core(coords, geom, tendon_data)
    debug_info = None
    if debug:
        debug_info = {
            "DFT_L_s": core.DFT_L_s,
            "DFT_state_A": core.DFT_state_A,
            "DFT_state_B": core.DFT_state_B,
            "DFT_state_C": core.DFT_state_C,
            "DFT_state_D": core.DFT_state_D,
            "DFT_L_s_A": core.DFT_L_s_A,
            "DFT_L_s_B": core.DFT_L_s_B,
            "DFT_L_s_C": core.DFT_L_s_C,
            "DFT_L_s_D": core.DFT_L_s_D,
            "DFT_q5_D": core.DFT_q5_D,
            "DFT_q6_B": core.DFT_q6_B,
            "DFT_delta_L_s": core.delta_l,
        }
    return TendonLengthOutput(delta_l=core.delta_l, debug=debug_info)
