# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import NamedTuple

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.geometry.common import TendonLengthOutput
from isaaclab.tendons.models.analytic.geometry.kinematics import TendonCoordinates
from isaaclab.tendons.models.analytic.geometry.shared import SharedTendonGeometry
from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


class EDT1DeltaCoreOutput(NamedTuple):
    delta_l: torch.Tensor
    EDT1_L_s: torch.Tensor
    EDT1_state_A: torch.Tensor
    EDT1_state_B: torch.Tensor


@torch.jit.script
def compute_edt1_delta_l_core(
    coords: TendonCoordinates,
    geom: SharedTendonGeometry,
    tendon_data: TendonDataJIT,
) -> EDT1DeltaCoreOutput:
    """EDT1 spring-length delta shared by debug and JIT paths."""
    EDT1_state_B = geom.EDT1_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
    EDT1_state_A = ~EDT1_state_B

    EDT1_L_s = torch.where(
        EDT1_state_B,
        geom.EDT1_l_cc,
        geom.EDT1_l_c5_A
        + geom.EDT1_q5_A * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT1_5C],
    )
    EDT1_delta_L_s = tendon_data.edt1_length - EDT1_L_s

    return EDT1DeltaCoreOutput(
        delta_l=EDT1_delta_L_s,
        EDT1_L_s=EDT1_L_s,
        EDT1_state_A=EDT1_state_A,
        EDT1_state_B=EDT1_state_B,
    )


def compute_edt1_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    core = compute_edt1_delta_l_core(coords, geom, tendon_data)
    state = {"EDT1_state_a": core.EDT1_state_A, "EDT1_state_b": core.EDT1_state_B}
    debug_info = None
    if debug:
        debug_info = {
            **state,
            "EDT1_x_c5": geom.EDT1_x_c5,
            "EDT1_phi_4_a": geom.EDT1_phi_4_a,
            "EDT1_thetahat_5_a": geom.EDT1_thetahat_5_a,
            "EDT1_l_c5_A": geom.EDT1_l_c5_A,
            "EDT1_phi_45_A": geom.EDT1_phi_45_A,
            "EDT1_q5_A": geom.EDT1_q5_A,
            "EDT1_thetahat_5_b": geom.EDT1_thetahat_5_b,
            "EDT1_phi_4_b": geom.EDT1_phi_4_b,
            "EDT1_h5_B": geom.EDT1_h5_B,
            "EDT1_l_cc": geom.EDT1_l_cc,
            "EDT1_L_s": core.EDT1_L_s,
            "EDT1_delta_L_s": core.delta_l,
        }
    return TendonLengthOutput(delta_l=core.delta_l, length=core.EDT1_L_s, state=state, debug=debug_info)
