from __future__ import annotations

from typing import NamedTuple

import torch

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.geometry.common import TendonLengthOutput
from isaaclab.tendons.models.analytic.geometry.kinematics import TendonCoordinates
from isaaclab.tendons.models.analytic.geometry.shared import SharedTendonGeometry
from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


class EDT2DeltaCoreOutput(NamedTuple):
    delta_l: torch.Tensor
    EDT2_L_s: torch.Tensor
    EDT2_state_A: torch.Tensor
    EDT2_state_B: torch.Tensor
    EDT2_state_C: torch.Tensor
    EDT2_state_D: torch.Tensor
    EDT2_L_s_A: torch.Tensor
    EDT2_L_s_B: torch.Tensor
    EDT2_L_s_C: torch.Tensor
    EDT2_L_s_D: torch.Tensor


@torch.jit.script
def compute_edt2_delta_l_core(
    coords: TendonCoordinates,
    geom: SharedTendonGeometry,
    tendon_data: TendonDataJIT,
) -> EDT2DeltaCoreOutput:
    """EDT2 spring-length delta shared by debug and JIT paths."""
    qhats = coords.qhats

    EDT2_L_s_A = (
        geom.EDT2_l_c5_A
        + geom.EDT2_q5_A * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT2_56]
        + qhats[:, tids.I_QHAT_EDT2_6] * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT2_6C]
    )
    EDT2_L_s_B = (
        geom.EDT2_l_c6_B
        + geom.EDT2_q6_B * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_EDT2_6C]
    )
    EDT2_L_s_C = geom.EDT2_l_cc_C
    EDT2_L_s_D = (
        geom.EDT2_l_c5_A + geom.EDT2_q5_D * tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5] + geom.EDT2_l_5c_D
    )

    EDT2_h5_B_disengaged = geom.EDT2_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
    EDT2_h5_C_disengaged = geom.EDT2_h5_C > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5]
    EDT2_h6_C_disengaged = geom.EDT2_h6_C > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]
    EDT2_h6_D_disengaged = geom.EDT2_h6_D > tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6]

    EDT2_state_C = (EDT2_h5_B_disengaged & EDT2_h6_C_disengaged) | (EDT2_h6_D_disengaged & EDT2_h5_C_disengaged)
    EDT2_state_B = ~EDT2_state_C & EDT2_h5_B_disengaged
    EDT2_state_D = ~EDT2_state_C & EDT2_h6_D_disengaged
    EDT2_state_A = ~(EDT2_state_B | EDT2_state_C | EDT2_state_D)

    EDT2_L_s = torch.where(
        EDT2_state_A,
        EDT2_L_s_A,
        torch.where(EDT2_state_B, EDT2_L_s_B, torch.where(EDT2_state_C, EDT2_L_s_C, EDT2_L_s_D)),
    )
    EDT2_delta_L_s = tendon_data.edt2_length - EDT2_L_s

    return EDT2DeltaCoreOutput(
        delta_l=EDT2_delta_L_s,
        EDT2_L_s=EDT2_L_s,
        EDT2_state_A=EDT2_state_A,
        EDT2_state_B=EDT2_state_B,
        EDT2_state_C=EDT2_state_C,
        EDT2_state_D=EDT2_state_D,
        EDT2_L_s_A=EDT2_L_s_A,
        EDT2_L_s_B=EDT2_L_s_B,
        EDT2_L_s_C=EDT2_L_s_C,
        EDT2_L_s_D=EDT2_L_s_D,
    )


def compute_edt2_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    core = compute_edt2_delta_l_core(coords, geom, tendon_data)
    state = {
        "EDT2_state_a": core.EDT2_state_A,
        "EDT2_state_b": core.EDT2_state_B,
        "EDT2_state_c": core.EDT2_state_C,
        "EDT2_state_d": core.EDT2_state_D,
    }
    debug_info = None
    if debug:
        debug_info = {
            **state,
            "EDT2_x_c5": geom.EDT2_x_c5,
            "EDT2_phi_4_a": geom.EDT2_phi_4_a,
            "EDT2_thetahat_5_a": geom.EDT2_thetahat_5_a,
            "EDT2_l_c5_A": geom.EDT2_l_c5_A,
            "EDT2_phi_45_A": geom.EDT2_phi_45_A,
            "EDT2_q5_A": geom.EDT2_q5_A,
            "EDT2_x_64prime": geom.EDT2_x_64prime,
            "EDT2_phi_6_a": geom.EDT2_phi_6_a,
            "EDT2_thetahat_4_a": geom.EDT2_thetahat_4_a,
            "EDT2_thetahat_4_b": geom.EDT2_thetahat_4_b,
            "EDT2_x_6c": geom.EDT2_x_6c,
            "EDT2_phi_6_d": geom.EDT2_phi_6_d,
            "EDT2_l_c6_B": geom.EDT2_l_c6_B,
            "EDT2_phi_6_c": geom.EDT2_phi_6_c,
            "EDT2_phi_6_B": geom.EDT2_phi_6_B,
            "EDT2_q6_B": geom.EDT2_q6_B,
            "EDT2_h5_B": geom.EDT2_h5_B,
            "EDT2_l_46_j": geom.EDT2_l_46_j,
            "EDT2_gamma_4": geom.EDT2_gamma_4,
            "EDT2_gamma_6": geom.EDT2_gamma_6,
            "EDT2_thetatilde_4": geom.EDT2_thetatilde_4,
            "EDT2_x_c6": geom.EDT2_x_c6,
            "EDT2_phi_4_b": geom.EDT2_phi_4_b,
            "EDT2_thetatilde_6": geom.EDT2_thetatilde_6,
            "EDT2_thetatilde_6_a": geom.EDT2_thetatilde_6_a,
            "EDT2_thetatilde_6_b": geom.EDT2_thetatilde_6_b,
            "EDT2_l_cc_C": geom.EDT2_l_cc_C,
            "EDT2_phi_4_d": geom.EDT2_phi_4_d,
            "EDT2_h6_C": geom.EDT2_h6_C,
            "EDT2_h5_C": geom.EDT2_h5_C,
            "EDT2_x_56": geom.EDT2_x_56,
            "EDT2_l_5c_D": geom.EDT2_l_5c_D,
            "EDT2_phi_56_a": geom.EDT2_phi_56_a,
            "EDT2_phi_56_b": geom.EDT2_phi_56_b,
            "EDT2_phi_56": geom.EDT2_phi_56,
            "EDT2_q5_D": geom.EDT2_q5_D,
            "EDT2_phi_7_D": geom.EDT2_phi_7_D,
            "EDT2_h6_D": geom.EDT2_h6_D,
            "EDT2_L_s_A": core.EDT2_L_s_A,
            "EDT2_L_s_B": core.EDT2_L_s_B,
            "EDT2_L_s_C": core.EDT2_L_s_C,
            "EDT2_L_s_D": core.EDT2_L_s_D,
            "EDT2_L_s": core.EDT2_L_s,
            "EDT2_delta_L_s": core.delta_l,
        }
    return TendonLengthOutput(delta_l=core.delta_l, length=core.EDT2_L_s, state=state, debug=debug_info)
