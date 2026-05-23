from __future__ import annotations

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.geometry.common import TendonLengthOutput


def compute_dft_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    del geom
    qs = coords.qs
    DFT_q5 = qs[:, tids.I_Q_DFT_5]
    DFT_q6 = qs[:, tids.I_Q_DFT_6]
    DFT_l_c5 = tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_C5]
    DFT_l_56 = tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_56]
    DFT_l_6c = tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_DFT_6C]

    DFT_delta_L_s = (
        tendon_data.dft_length
        - DFT_l_c5
        - DFT_q5 * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5]
        - DFT_l_56
        - DFT_q6 * tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6]
        - DFT_l_6c
    )

    debug_info = None
    if debug:
        debug_info = {
            "DFT_q5": DFT_q5,
            "DFT_q6": DFT_q6,
            "DFT_l_c5": DFT_l_c5,
            "DFT_l_56": DFT_l_56,
            "DFT_l_6c": DFT_l_6c,
            "DFT_delta_L_s": DFT_delta_L_s,
        }
    return TendonLengthOutput(delta_l=DFT_delta_L_s, debug=debug_info)
