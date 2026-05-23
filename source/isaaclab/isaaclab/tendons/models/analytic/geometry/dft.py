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
    DFT_q5: torch.Tensor
    DFT_q6: torch.Tensor
    DFT_l_c5: torch.Tensor
    DFT_l_56: torch.Tensor
    DFT_l_6c: torch.Tensor


@torch.jit.script
def compute_dft_delta_l_core(
    coords: TendonCoordinates,
    geom: SharedTendonGeometry,
    tendon_data: TendonDataJIT,
) -> DFTDeltaCoreOutput:
    """DFT spring-length delta shared by debug and JIT paths."""
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

    return DFTDeltaCoreOutput(
        delta_l=DFT_delta_L_s,
        DFT_q5=DFT_q5,
        DFT_q6=DFT_q6,
        DFT_l_c5=DFT_l_c5,
        DFT_l_56=DFT_l_56,
        DFT_l_6c=DFT_l_6c,
    )


def compute_dft_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    core = compute_dft_delta_l_core(coords, geom, tendon_data)
    debug_info = None
    if debug:
        debug_info = {
            "DFT_q5": core.DFT_q5,
            "DFT_q6": core.DFT_q6,
            "DFT_l_c5": core.DFT_l_c5,
            "DFT_l_56": core.DFT_l_56,
            "DFT_l_6c": core.DFT_l_6c,
            "DFT_delta_L_s": core.delta_l,
        }
    return TendonLengthOutput(delta_l=core.delta_l, debug=debug_info)
