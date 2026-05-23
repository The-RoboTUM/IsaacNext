from __future__ import annotations

import torch

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.geometry.common import TendonLengthOutput


def compute_edt1_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
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

    state = {"EDT1_state_a": EDT1_state_A, "EDT1_state_b": EDT1_state_B}
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
            "EDT1_L_s": EDT1_L_s,
            "EDT1_delta_L_s": EDT1_delta_L_s,
        }
    return TendonLengthOutput(delta_l=EDT1_delta_L_s, length=EDT1_L_s, state=state, debug=debug_info)
