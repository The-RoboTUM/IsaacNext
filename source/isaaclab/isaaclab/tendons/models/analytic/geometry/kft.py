from __future__ import annotations

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.geometry.common import TendonLengthOutput


def compute_kft_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    del coords
    KFT_delta_L_s = tendon_data.kft_length - geom.KFT_q8 * tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8] - geom.KFT_l_8c
    debug_info = None
    if debug:
        debug_info = {
            "KFT_l_8c": geom.KFT_l_8c,
            "KFT_phi_8": geom.KFT_phi_8,
            "KFT_phi_8_a": geom.KFT_phi_8_a,
            "KFT_q8": geom.KFT_q8,
            "KFT_delta_L_s": KFT_delta_L_s,
        }
    return TendonLengthOutput(delta_l=KFT_delta_L_s, debug=debug_info)
