# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import isaaclab.tendons.models.analytic_jax.indices as tids
from isaaclab.tendons.models.analytic_jax.geometry.common import TendonLengthOutput
from isaaclab.tendons.models.analytic_jax.geometry.kinematics import TendonCoordinates
from isaaclab.tendons.models.analytic_jax.geometry.shared import SharedTendonGeometry
from isaaclab.tendons.models.analytic_jax.tendon_data import TendonDataJIT


class KFTDeltaCoreOutput(NamedTuple):
    delta_l: jnp.ndarray


def compute_kft_delta_l_core(
    coords: TendonCoordinates,
    geom: SharedTendonGeometry,
    tendon_data: TendonDataJIT,
) -> KFTDeltaCoreOutput:
    """KFT spring-length delta shared by debug and JIT paths."""
    KFT_delta_L_s = (
        tendon_data.kft_length - geom.KFT_q8 * tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8] - geom.KFT_l_8c
    )
    return KFTDeltaCoreOutput(delta_l=KFT_delta_L_s)


def compute_kft_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    core = compute_kft_delta_l_core(coords, geom, tendon_data)
    debug_info = None
    if debug:
        debug_info = {
            "KFT_l_8c": geom.KFT_l_8c,
            "KFT_phi_8": geom.KFT_phi_8,
            "KFT_phi_8_a": geom.KFT_phi_8_a,
            "KFT_q8": geom.KFT_q8,
            "KFT_delta_L_s": core.delta_l,
        }
    return TendonLengthOutput(delta_l=core.delta_l, debug=debug_info)
