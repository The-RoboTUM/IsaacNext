# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import isaaclab.tendons.models.analytic_jax.indices as tids
from isaaclab.tendons.models.analytic_jax.geometry.common import angle_from_sws
from isaaclab.tendons.models.analytic_jax.geometry.kinematics import TendonCoordinates
from isaaclab.tendons.models.analytic_jax.tendon_data import TendonDataJIT


def _safe_sqrt(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.clip(value, min=1.0e-12))


def _safe_acos(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.arccos(jnp.clip(value, -1.0 + 1.0e-6, 1.0 - 1.0e-6))


class SharedTendonGeometry(NamedTuple):
    """Intermediate geometric terms computed once and reused by tendon calculators.

    This NamedTuple is the single geometry payload used by both the eager debug
    wrappers and the TorchScript path. Field names preserve the original
    variable names to keep validation against old debug logs straightforward.
    """

    # GST shared terms
    GST_x_4prime6_squared: jnp.ndarray
    GST_x_4prime6: jnp.ndarray
    GST_l_4prime6_squared: jnp.ndarray
    GST_l_4prime6: jnp.ndarray
    GST_phi_4prime_a: jnp.ndarray
    GST_phi_4prime_b: jnp.ndarray
    GST_phi_4prime_B: jnp.ndarray
    GST_h5_B: jnp.ndarray
    GST_theta_6_a: jnp.ndarray
    GST_theta_6_b: jnp.ndarray
    GST_x_4prime7_squared: jnp.ndarray
    GST_x_4prime7: jnp.ndarray
    GST_phi_4prime_d: jnp.ndarray
    GST_phi_4prime_c: jnp.ndarray
    GST_phi_4prime_C: jnp.ndarray
    GST_h5_C: jnp.ndarray
    GST_h6_C: jnp.ndarray
    GST_x_57_squared: jnp.ndarray
    GST_x_57: jnp.ndarray
    GST_l_57_squared: jnp.ndarray
    GST_l_57: jnp.ndarray
    GST_phi_5_a: jnp.ndarray
    GST_phi_5_b: jnp.ndarray
    GST_phi_5_D: jnp.ndarray
    GST_h6_D: jnp.ndarray
    # KFT shared terms
    KFT_l_8c_j_squared: jnp.ndarray
    KFT_l_8c: jnp.ndarray
    KFT_phi_8: jnp.ndarray
    KFT_phi_8_a: jnp.ndarray
    KFT_q8: jnp.ndarray
    # DFT shared terms
    DFT_phi_4_a: jnp.ndarray
    DFT_x_c6_squared: jnp.ndarray
    DFT_x_c6: jnp.ndarray
    DFT_l_c6_squared: jnp.ndarray
    DFT_l_c6: jnp.ndarray
    DFT_phi_4_b: jnp.ndarray
    DFT_phi_4_B: jnp.ndarray
    DFT_phi_6_B: jnp.ndarray
    DFT_q_6_B: jnp.ndarray
    DFT_h5_B: jnp.ndarray
    DFT_theta_6_a: jnp.ndarray
    DFT_theta_6_b: jnp.ndarray
    DFT_l_c7_squared: jnp.ndarray
    DFT_phi_4_d: jnp.ndarray
    DFT_phi_4_C: jnp.ndarray
    DFT_h5_C: jnp.ndarray
    DFT_h6_C: jnp.ndarray
    DFT_x_57_squared: jnp.ndarray
    DFT_x_57: jnp.ndarray
    DFT_l_57_squared: jnp.ndarray
    DFT_l_57: jnp.ndarray
    DFT_phi_5_a: jnp.ndarray
    DFT_phi_5_b: jnp.ndarray
    DFT_phi_5_D: jnp.ndarray
    DFT_h6_D: jnp.ndarray
    # EDT1 shared terms
    EDT1_x_c5_squared: jnp.ndarray
    EDT1_x_c5: jnp.ndarray
    EDT1_phi_4_a: jnp.ndarray
    EDT1_thetahat_5_a: jnp.ndarray
    EDT1_l_c5_A: jnp.ndarray
    EDT1_phi_45_A: jnp.ndarray
    EDT1_q5_A: jnp.ndarray
    EDT1_thetahat_5_b: jnp.ndarray
    EDT1_phi_4_b: jnp.ndarray
    EDT1_h5_B: jnp.ndarray
    EDT1_l_cc: jnp.ndarray
    # EDT2 shared terms
    EDT2_x_c5_squared: jnp.ndarray
    EDT2_x_c5: jnp.ndarray
    EDT2_phi_4_a: jnp.ndarray
    EDT2_thetahat_5_a: jnp.ndarray
    EDT2_l_c5_A: jnp.ndarray
    EDT2_phi_45_A: jnp.ndarray
    EDT2_q5_A: jnp.ndarray
    EDT2_x_64prime_squared: jnp.ndarray
    EDT2_x_64prime: jnp.ndarray
    EDT2_phi_6_a: jnp.ndarray
    EDT2_thetahat_4_a: jnp.ndarray
    EDT2_thetahat_4_b: jnp.ndarray
    EDT2_x_6c_squared: jnp.ndarray
    EDT2_x_6c: jnp.ndarray
    EDT2_phi_6_d: jnp.ndarray
    EDT2_l_c6_B: jnp.ndarray
    EDT2_phi_6_c: jnp.ndarray
    EDT2_phi_6_B: jnp.ndarray
    EDT2_q6_B: jnp.ndarray
    EDT2_h5_B: jnp.ndarray
    EDT2_l_46_j_squared: jnp.ndarray
    EDT2_l_46_j: jnp.ndarray
    EDT2_gamma_4: jnp.ndarray
    EDT2_gamma_6: jnp.ndarray
    EDT2_thetatilde_4: jnp.ndarray
    EDT2_x_c6_squared: jnp.ndarray
    EDT2_x_c6: jnp.ndarray
    EDT2_phi_4_b: jnp.ndarray
    EDT2_thetatilde_6: jnp.ndarray
    EDT2_thetatilde_6_a: jnp.ndarray
    EDT2_thetatilde_6_b: jnp.ndarray
    EDT2_l_cc_squared: jnp.ndarray
    EDT2_l_cc_C: jnp.ndarray
    EDT2_phi_4_d: jnp.ndarray
    EDT2_h6_C: jnp.ndarray
    EDT2_h5_C: jnp.ndarray
    EDT2_x_56_squared: jnp.ndarray
    EDT2_x_56: jnp.ndarray
    EDT2_l_5c_D: jnp.ndarray
    EDT2_phi_56_a: jnp.ndarray
    EDT2_phi_56_b: jnp.ndarray
    EDT2_phi_56: jnp.ndarray
    EDT2_q5_D: jnp.ndarray
    EDT2_phi_7_D: jnp.ndarray
    EDT2_h6_D: jnp.ndarray


def shared_geometry_as_debug_dict(
    geom: SharedTendonGeometry,
) -> dict[str, jnp.ndarray]:
    return {
        "GST_x_4prime6_squared": geom.GST_x_4prime6_squared,
        "GST_x_4prime6": geom.GST_x_4prime6,
        "GST_l_4prime6_squared": geom.GST_l_4prime6_squared,
        "GST_l_4prime6": geom.GST_l_4prime6,
        "GST_phi_4prime_a": geom.GST_phi_4prime_a,
        "GST_phi_4prime_b": geom.GST_phi_4prime_b,
        "GST_phi_4prime_B": geom.GST_phi_4prime_B,
        "GST_h5_B": geom.GST_h5_B,
        "GST_theta_6_a": geom.GST_theta_6_a,
        "GST_theta_6_b": geom.GST_theta_6_b,
        "GST_x_4prime7_squared": geom.GST_x_4prime7_squared,
        "GST_x_4prime7": geom.GST_x_4prime7,
        "GST_phi_4prime_d": geom.GST_phi_4prime_d,
        "GST_phi_4prime_c": geom.GST_phi_4prime_c,
        "GST_phi_4prime_C": geom.GST_phi_4prime_C,
        "GST_h5_C": geom.GST_h5_C,
        "GST_h6_C": geom.GST_h6_C,
        "GST_x_57_squared": geom.GST_x_57_squared,
        "GST_x_57": geom.GST_x_57,
        "GST_l_57_squared": geom.GST_l_57_squared,
        "GST_l_57": geom.GST_l_57,
        "GST_phi_5_a": geom.GST_phi_5_a,
        "GST_phi_5_b": geom.GST_phi_5_b,
        "GST_phi_5_D": geom.GST_phi_5_D,
        "GST_h6_D": geom.GST_h6_D,
        "KFT_l_8c_j_squared": geom.KFT_l_8c_j_squared,
        "KFT_l_8c": geom.KFT_l_8c,
        "KFT_phi_8": geom.KFT_phi_8,
        "KFT_phi_8_a": geom.KFT_phi_8_a,
        "KFT_q8": geom.KFT_q8,
        "DFT_phi_4_a": geom.DFT_phi_4_a,
        "DFT_x_c6_squared": geom.DFT_x_c6_squared,
        "DFT_x_c6": geom.DFT_x_c6,
        "DFT_l_c6_squared": geom.DFT_l_c6_squared,
        "DFT_l_c6": geom.DFT_l_c6,
        "DFT_phi_4_b": geom.DFT_phi_4_b,
        "DFT_phi_4_B": geom.DFT_phi_4_B,
        "DFT_phi_6_B": geom.DFT_phi_6_B,
        "DFT_q_6_B": geom.DFT_q_6_B,
        "DFT_h5_B": geom.DFT_h5_B,
        "DFT_theta_6_a": geom.DFT_theta_6_a,
        "DFT_theta_6_b": geom.DFT_theta_6_b,
        "DFT_l_c7_squared": geom.DFT_l_c7_squared,
        "DFT_phi_4_d": geom.DFT_phi_4_d,
        "DFT_phi_4_C": geom.DFT_phi_4_C,
        "DFT_h5_C": geom.DFT_h5_C,
        "DFT_h6_C": geom.DFT_h6_C,
        "DFT_x_57_squared": geom.DFT_x_57_squared,
        "DFT_x_57": geom.DFT_x_57,
        "DFT_l_57_squared": geom.DFT_l_57_squared,
        "DFT_l_57": geom.DFT_l_57,
        "DFT_phi_5_a": geom.DFT_phi_5_a,
        "DFT_phi_5_b": geom.DFT_phi_5_b,
        "DFT_phi_5_D": geom.DFT_phi_5_D,
        "DFT_h6_D": geom.DFT_h6_D,
        "EDT1_x_c5_squared": geom.EDT1_x_c5_squared,
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
        "EDT2_x_c5_squared": geom.EDT2_x_c5_squared,
        "EDT2_x_c5": geom.EDT2_x_c5,
        "EDT2_phi_4_a": geom.EDT2_phi_4_a,
        "EDT2_thetahat_5_a": geom.EDT2_thetahat_5_a,
        "EDT2_l_c5_A": geom.EDT2_l_c5_A,
        "EDT2_phi_45_A": geom.EDT2_phi_45_A,
        "EDT2_q5_A": geom.EDT2_q5_A,
        "EDT2_x_64prime_squared": geom.EDT2_x_64prime_squared,
        "EDT2_x_64prime": geom.EDT2_x_64prime,
        "EDT2_phi_6_a": geom.EDT2_phi_6_a,
        "EDT2_thetahat_4_a": geom.EDT2_thetahat_4_a,
        "EDT2_thetahat_4_b": geom.EDT2_thetahat_4_b,
        "EDT2_x_6c_squared": geom.EDT2_x_6c_squared,
        "EDT2_x_6c": geom.EDT2_x_6c,
        "EDT2_phi_6_d": geom.EDT2_phi_6_d,
        "EDT2_l_c6_B": geom.EDT2_l_c6_B,
        "EDT2_phi_6_c": geom.EDT2_phi_6_c,
        "EDT2_phi_6_B": geom.EDT2_phi_6_B,
        "EDT2_q6_B": geom.EDT2_q6_B,
        "EDT2_h5_B": geom.EDT2_h5_B,
        "EDT2_l_46_j_squared": geom.EDT2_l_46_j_squared,
        "EDT2_l_46_j": geom.EDT2_l_46_j,
        "EDT2_gamma_4": geom.EDT2_gamma_4,
        "EDT2_gamma_6": geom.EDT2_gamma_6,
        "EDT2_thetatilde_4": geom.EDT2_thetatilde_4,
        "EDT2_x_c6_squared": geom.EDT2_x_c6_squared,
        "EDT2_x_c6": geom.EDT2_x_c6,
        "EDT2_phi_4_b": geom.EDT2_phi_4_b,
        "EDT2_thetatilde_6": geom.EDT2_thetatilde_6,
        "EDT2_thetatilde_6_a": geom.EDT2_thetatilde_6_a,
        "EDT2_thetatilde_6_b": geom.EDT2_thetatilde_6_b,
        "EDT2_l_cc_squared": geom.EDT2_l_cc_squared,
        "EDT2_l_cc_C": geom.EDT2_l_cc_C,
        "EDT2_phi_4_d": geom.EDT2_phi_4_d,
        "EDT2_h6_C": geom.EDT2_h6_C,
        "EDT2_h5_C": geom.EDT2_h5_C,
        "EDT2_x_56_squared": geom.EDT2_x_56_squared,
        "EDT2_x_56": geom.EDT2_x_56,
        "EDT2_l_5c_D": geom.EDT2_l_5c_D,
        "EDT2_phi_56_a": geom.EDT2_phi_56_a,
        "EDT2_phi_56_b": geom.EDT2_phi_56_b,
        "EDT2_phi_56": geom.EDT2_phi_56,
        "EDT2_q5_D": geom.EDT2_q5_D,
        "EDT2_phi_7_D": geom.EDT2_phi_7_D,
        "EDT2_h6_D": geom.EDT2_h6_D,
    }


def compute_shared_tendon_geometry(coords: TendonCoordinates, tendon_data: TendonDataJIT) -> SharedTendonGeometry:
    """Compute all reusable geometry terms once before tendon length calculations."""
    thetas = coords.thetas
    theta_hats = coords.theta_hats

    # ---------------- GST ----------------
    GST_x_4prime6_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * jnp.cos(thetas[:, tids.I_THETA_GST_5])
    )
    GST_x_4prime6 = _safe_sqrt(GST_x_4prime6_squared)
    GST_l_4prime6_squared = (
        GST_x_4prime6_squared
        - (tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6])
        ** 2
    )
    GST_l_4prime6 = _safe_sqrt(GST_l_4prime6_squared)
    GST_phi_4prime_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.I_THETA_GST_5],
    )
    GST_phi_4prime_b = _safe_acos(
        (
            tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_4prime]
            + GST_x_4prime6_squared
            - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_6]
            - GST_l_4prime6_squared
        )
        / (2 * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] * GST_x_4prime6)
    )
    GST_phi_4prime_B = GST_phi_4prime_a + GST_phi_4prime_b
    GST_h5_B = tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] - tendon_data.link_lengths[
        :, tids.I_LINK_4prime5
    ] * jnp.cos(GST_phi_4prime_B)

    GST_theta_6_a = jnp.pi - thetas[:, tids.I_THETA_GST_5] - GST_phi_4prime_a
    GST_theta_6_b = thetas[:, tids.I_THETA_ALL_6] - GST_theta_6_a
    GST_x_4prime7_squared = (
        GST_x_4prime6_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2 * GST_x_4prime6 * tendon_data.link_lengths[:, tids.I_LINK_67] * jnp.cos(GST_theta_6_b)
    )
    GST_x_4prime7 = _safe_sqrt(GST_x_4prime7_squared)
    GST_phi_4prime_d = angle_from_sws(GST_x_4prime6, tendon_data.link_lengths[:, tids.I_LINK_67], GST_theta_6_b)
    GST_phi_4prime_c = _safe_acos(tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] / GST_x_4prime7)
    GST_phi_4prime_C = GST_phi_4prime_a + GST_phi_4prime_c + GST_phi_4prime_d
    GST_h5_C = tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] - tendon_data.link_lengths[
        :, tids.I_LINK_4prime5
    ] * jnp.cos(GST_phi_4prime_C)
    GST_h6_C = tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] - GST_x_4prime6 * jnp.cos(
        GST_phi_4prime_c + GST_phi_4prime_d
    )

    GST_x_57_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * tendon_data.link_lengths[:, tids.I_LINK_67]
        * jnp.cos(thetas[:, tids.I_THETA_ALL_6])
    )
    GST_x_57 = _safe_sqrt(GST_x_57_squared)
    GST_l_57_squared = GST_x_57_squared - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] ** 2
    GST_l_57 = _safe_sqrt(GST_l_57_squared)
    GST_phi_5_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_67],
        thetas[:, tids.I_THETA_ALL_6],
    )
    GST_phi_5_b = _safe_acos(tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] / GST_x_57)
    GST_phi_5_D = GST_phi_5_a + GST_phi_5_b
    GST_h6_D = tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] - tendon_data.link_lengths[
        :, tids.I_LINK_56
    ] * jnp.cos(GST_phi_5_D)

    # ---------------- KFT ----------------
    KFT_l_8c_j_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_38]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_KFT_3C]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_38]
        * tendon_data.link_lengths[:, tids.I_LINK_KFT_3C]
        * jnp.cos(theta_hats[:, tids.I_THETA_KFT_3])
    )
    KFT_l_8c = _safe_sqrt(KFT_l_8c_j_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_KFT_8])
    KFT_phi_8 = jnp.arctan2(KFT_l_8c, tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8])
    KFT_phi_8_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_38],
        tendon_data.link_lengths[:, tids.I_LINK_KFT_3C],
        theta_hats[:, tids.I_THETA_KFT_3],
    )
    KFT_q8 = thetas[:, tids.I_THETA_KFT_8] - KFT_phi_8 + KFT_phi_8_a

    # ---------------- DFT -----------------
    # shared between state B and D
    DFT_phi_4_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_DFT_C5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.I_THETA_DFT_5],
    )
    DFT_x_c6_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_DFT_C5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_DFT_C5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * jnp.cos(thetas[:, tids.I_THETA_DFT_5])
    )
    DFT_x_c6 = _safe_sqrt(DFT_x_c6_squared)

    # state B
    DFT_l_c6_squared = DFT_x_c6_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_DFT_6]
    DFT_l_c6 = _safe_sqrt(DFT_l_c6_squared)
    DFT_phi_4_b = jnp.arctan2(tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6], DFT_l_c6)
    DFT_phi_4_B = DFT_phi_4_a + DFT_phi_4_b
    DFT_phi_6_B = jnp.pi * 1.5 - DFT_phi_4_B - thetas[:, tids.I_THETA_DFT_5]
    DFT_q_6_B = (
        thetas[:, tids.I_THETA_ALL_6]
        - DFT_phi_6_B
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_DFT_6C_J6]
    )
    DFT_h5_B = tendon_data.link_lengths[:, tids.I_LINK_DFT_C5] * jnp.sin(DFT_phi_4_B)

    # state C
    DFT_theta_6_a = jnp.pi - thetas[:, tids.I_THETA_DFT_5] - DFT_phi_4_a
    DFT_theta_6_b = thetas[:, tids.I_THETA_ALL_6] - DFT_theta_6_a
    DFT_l_c7_squared = (
        DFT_x_c6_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2 * DFT_x_c6 * tendon_data.link_lengths[:, tids.I_LINK_67] * jnp.cos(DFT_theta_6_b)
    )
    DFT_phi_4_d = angle_from_sws(DFT_x_c6, tendon_data.link_lengths[:, tids.I_LINK_67], DFT_theta_6_b)
    DFT_phi_4_C = DFT_phi_4_a + DFT_phi_4_d
    DFT_h5_C = tendon_data.link_lengths[:, tids.I_LINK_DFT_C5] * jnp.sin(DFT_phi_4_C)
    DFT_h6_C = DFT_x_c6 * jnp.sin(DFT_phi_4_d)

    # state D
    DFT_x_57_squared = GST_x_57_squared
    DFT_x_57 = GST_x_57
    DFT_l_57_squared = DFT_x_57_squared - tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5] ** 2
    DFT_l_57 = _safe_sqrt(DFT_l_57_squared)
    DFT_phi_5_a = GST_phi_5_a
    DFT_phi_5_b = _safe_acos(tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5] / DFT_x_57)
    DFT_phi_5_D = DFT_phi_5_a + DFT_phi_5_b
    DFT_h6_D = tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5] - tendon_data.link_lengths[
        :, tids.I_LINK_56
    ] * jnp.cos(DFT_phi_5_D)

    # ---------------- EDT1 ----------------
    EDT1_x_c5_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_C4]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4]
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * jnp.cos(theta_hats[:, tids.I_THETA_EDT1_4])
    )
    EDT1_x_c5 = _safe_sqrt(EDT1_x_c5_squared)
    EDT1_phi_4_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT1_4],
    )
    EDT1_thetahat_5_a = jnp.pi - theta_hats[:, tids.I_THETA_EDT1_4] - EDT1_phi_4_a
    EDT1_l_c5_A = _safe_sqrt(EDT1_x_c5_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT1_5])
    EDT1_phi_45_A = jnp.arctan2(EDT1_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5])
    EDT1_q5_A = (
        2 * jnp.pi
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_EDT1_5C_J5]
        - thetas[:, tids.I_THETA_EDT1_5]
        - EDT1_thetahat_5_a
        - EDT1_phi_45_A
    )
    EDT1_thetahat_5_b = theta_hats[:, tids.I_THETA_EDT1_5] - EDT1_thetahat_5_a
    EDT1_phi_4_b = angle_from_sws(EDT1_x_c5, tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C], EDT1_thetahat_5_b)
    EDT1_h5_B = EDT1_x_c5 * jnp.sin(EDT1_phi_4_b)
    EDT1_l_cc = _safe_sqrt(
        EDT1_x_c5_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_5C]
        - 2 * EDT1_x_c5 * tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C] * jnp.cos(EDT1_thetahat_5_b)
    )

    # ------------- EDT2 -------------
    EDT2_x_c5_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * jnp.cos(theta_hats[:, tids.I_THETA_EDT2_4])
    )
    EDT2_x_c5 = _safe_sqrt(EDT2_x_c5_squared)
    EDT2_phi_4_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT2_4],
    )
    EDT2_thetahat_5_a = jnp.pi - theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_phi_4_a

    # state A: tendon wraps around j5 and j6 pulleys
    EDT2_l_c5_A = _safe_sqrt(EDT2_x_c5_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5])
    EDT2_phi_45_A = jnp.arctan2(EDT2_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5])
    EDT2_q5_A = (
        2 * jnp.pi
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_EDT2_56_J5]
        - thetas[:, tids.I_THETA_EDT2_5]
        - EDT2_thetahat_5_a
        - EDT2_phi_45_A
    )

    # state B: tendon wraps around j6 pulley but not j5 pulley
    EDT2_x_64prime_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * jnp.cos(theta_hats[:, tids.I_THETA_EDT2_5])
    )
    EDT2_x_64prime = _safe_sqrt(EDT2_x_64prime_squared)
    EDT2_phi_6_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT2_5],
    )
    EDT2_thetahat_4_a = jnp.pi - theta_hats[:, tids.I_THETA_EDT2_5] - EDT2_phi_6_a
    EDT2_thetahat_4_b = theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_thetahat_4_a
    EDT2_x_6c_squared = (
        EDT2_x_64prime_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
        - 2 * EDT2_x_64prime * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4] * jnp.cos(EDT2_thetahat_4_b)
    )
    EDT2_x_6c = _safe_sqrt(EDT2_x_6c_squared)
    EDT2_phi_6_d = angle_from_sws(
        EDT2_x_64prime,
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        EDT2_thetahat_4_b,
    )
    EDT2_l_c6_B = _safe_sqrt(EDT2_x_6c_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_6])
    EDT2_phi_6_c = jnp.arctan2(EDT2_l_c6_B, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6])
    EDT2_phi_6_B = EDT2_phi_6_a + EDT2_phi_6_c + EDT2_phi_6_d
    EDT2_q6_B = (
        theta_hats[:, tids.I_THETA_ALL_6]
        - EDT2_phi_6_B
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_EDT2_67_J6]
    )
    EDT2_h5_B = tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6] - tendon_data.link_lengths[
        :, tids.I_LINK_56
    ] * jnp.cos(EDT2_phi_6_B)

    # state C: tendon does not wrap around any pulley
    EDT2_l_46_j_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * jnp.cos(thetas[:, tids.I_THETA_EDT2_5])
    )
    EDT2_l_46_j = _safe_sqrt(EDT2_l_46_j_squared)
    EDT2_gamma_4 = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.I_THETA_EDT2_5],
    )
    EDT2_gamma_6 = jnp.pi - EDT2_gamma_4 - thetas[:, tids.I_THETA_EDT2_5]
    EDT2_thetatilde_4 = theta_hats[:, tids.I_THETA_EDT2_4] + EDT2_gamma_4
    EDT2_x_c6_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
        + EDT2_l_46_j_squared
        - 2 * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4] * EDT2_l_46_j * jnp.cos(EDT2_thetatilde_4)
    )
    EDT2_x_c6 = _safe_sqrt(EDT2_x_c6_squared)
    EDT2_phi_4_b = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        EDT2_l_46_j,
        EDT2_thetatilde_4,
    )
    EDT2_thetatilde_6 = theta_hats[:, tids.I_THETA_ALL_6] + EDT2_gamma_6
    EDT2_thetatilde_6_a = jnp.pi - EDT2_thetatilde_4 - EDT2_phi_4_b
    EDT2_thetatilde_6_b = EDT2_thetatilde_6 - EDT2_thetatilde_6_a
    EDT2_l_cc_squared = (
        EDT2_x_c6_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2 * EDT2_x_c6 * tendon_data.link_lengths[:, tids.I_LINK_67] * jnp.cos(EDT2_thetatilde_6_b)
    )
    EDT2_l_cc_C = _safe_sqrt(EDT2_l_cc_squared)
    EDT2_phi_4_d = angle_from_sws(EDT2_x_c6, tendon_data.link_lengths[:, tids.I_LINK_67], EDT2_thetatilde_6_b)

    EDT2_h6_C = EDT2_x_c6 * jnp.sin(EDT2_phi_4_d)
    EDT2_h5_C = EDT2_h6_C - tendon_data.link_lengths[:, tids.I_LINK_56] * jnp.sin(EDT2_gamma_6)

    # state D: tendon wraps around j5 pulley but not j6 pulley
    EDT2_x_56_squared = (
        tendon_data.link_lengths_squared[:, tids.I_LINK_56]
        + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
        - 2
        * tendon_data.link_lengths[:, tids.I_LINK_56]
        * tendon_data.link_lengths[:, tids.I_LINK_67]
        * jnp.cos(theta_hats[:, tids.I_THETA_ALL_6])
    )
    EDT2_x_56 = _safe_sqrt(EDT2_x_56_squared)
    EDT2_l_5c_D = _safe_sqrt(EDT2_x_56_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5])
    EDT2_phi_56_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_67],
        theta_hats[:, tids.I_THETA_ALL_6],
    )
    EDT2_phi_56_b = jnp.arctan2(EDT2_l_5c_D, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5])
    EDT2_phi_56 = EDT2_phi_56_a + EDT2_phi_56_b
    EDT2_q5_D = (
        2 * jnp.pi
        - EDT2_phi_56
        - thetas[:, tids.I_THETA_EDT2_5]
        - EDT2_thetahat_5_a
        - EDT2_phi_45_A  # note: phi_45 is the same for states A and D
    )
    EDT2_phi_7_D = 1.5 * jnp.pi - theta_hats[:, tids.I_THETA_ALL_6] - EDT2_phi_56
    EDT2_h6_D = tendon_data.link_lengths[:, tids.I_LINK_67] * jnp.sin(EDT2_phi_7_D)

    return SharedTendonGeometry(
        GST_x_4prime6_squared=GST_x_4prime6_squared,
        GST_x_4prime6=GST_x_4prime6,
        GST_l_4prime6_squared=GST_l_4prime6_squared,
        GST_l_4prime6=GST_l_4prime6,
        GST_phi_4prime_a=GST_phi_4prime_a,
        GST_phi_4prime_b=GST_phi_4prime_b,
        GST_phi_4prime_B=GST_phi_4prime_B,
        GST_h5_B=GST_h5_B,
        GST_theta_6_a=GST_theta_6_a,
        GST_theta_6_b=GST_theta_6_b,
        GST_x_4prime7_squared=GST_x_4prime7_squared,
        GST_x_4prime7=GST_x_4prime7,
        GST_phi_4prime_d=GST_phi_4prime_d,
        GST_phi_4prime_c=GST_phi_4prime_c,
        GST_phi_4prime_C=GST_phi_4prime_C,
        GST_h5_C=GST_h5_C,
        GST_h6_C=GST_h6_C,
        GST_x_57_squared=GST_x_57_squared,
        GST_x_57=GST_x_57,
        GST_l_57_squared=GST_l_57_squared,
        GST_l_57=GST_l_57,
        GST_phi_5_a=GST_phi_5_a,
        GST_phi_5_b=GST_phi_5_b,
        GST_phi_5_D=GST_phi_5_D,
        GST_h6_D=GST_h6_D,
        KFT_l_8c_j_squared=KFT_l_8c_j_squared,
        KFT_l_8c=KFT_l_8c,
        KFT_phi_8=KFT_phi_8,
        KFT_phi_8_a=KFT_phi_8_a,
        KFT_q8=KFT_q8,
        DFT_phi_4_a=DFT_phi_4_a,
        DFT_x_c6_squared=DFT_x_c6_squared,
        DFT_x_c6=DFT_x_c6,
        DFT_l_c6_squared=DFT_l_c6_squared,
        DFT_l_c6=DFT_l_c6,
        DFT_phi_4_b=DFT_phi_4_b,
        DFT_phi_4_B=DFT_phi_4_B,
        DFT_phi_6_B=DFT_phi_6_B,
        DFT_q_6_B=DFT_q_6_B,
        DFT_h5_B=DFT_h5_B,
        DFT_theta_6_a=DFT_theta_6_a,
        DFT_theta_6_b=DFT_theta_6_b,
        DFT_l_c7_squared=DFT_l_c7_squared,
        DFT_phi_4_d=DFT_phi_4_d,
        DFT_phi_4_C=DFT_phi_4_C,
        DFT_h5_C=DFT_h5_C,
        DFT_h6_C=DFT_h6_C,
        DFT_x_57_squared=DFT_x_57_squared,
        DFT_x_57=DFT_x_57,
        DFT_l_57_squared=DFT_l_57_squared,
        DFT_l_57=DFT_l_57,
        DFT_phi_5_a=DFT_phi_5_a,
        DFT_phi_5_b=DFT_phi_5_b,
        DFT_phi_5_D=DFT_phi_5_D,
        DFT_h6_D=DFT_h6_D,
        EDT1_x_c5_squared=EDT1_x_c5_squared,
        EDT1_x_c5=EDT1_x_c5,
        EDT1_phi_4_a=EDT1_phi_4_a,
        EDT1_thetahat_5_a=EDT1_thetahat_5_a,
        EDT1_l_c5_A=EDT1_l_c5_A,
        EDT1_phi_45_A=EDT1_phi_45_A,
        EDT1_q5_A=EDT1_q5_A,
        EDT1_thetahat_5_b=EDT1_thetahat_5_b,
        EDT1_phi_4_b=EDT1_phi_4_b,
        EDT1_h5_B=EDT1_h5_B,
        EDT1_l_cc=EDT1_l_cc,
        EDT2_x_c5_squared=EDT2_x_c5_squared,
        EDT2_x_c5=EDT2_x_c5,
        EDT2_phi_4_a=EDT2_phi_4_a,
        EDT2_thetahat_5_a=EDT2_thetahat_5_a,
        EDT2_l_c5_A=EDT2_l_c5_A,
        EDT2_phi_45_A=EDT2_phi_45_A,
        EDT2_q5_A=EDT2_q5_A,
        EDT2_x_64prime_squared=EDT2_x_64prime_squared,
        EDT2_x_64prime=EDT2_x_64prime,
        EDT2_phi_6_a=EDT2_phi_6_a,
        EDT2_thetahat_4_a=EDT2_thetahat_4_a,
        EDT2_thetahat_4_b=EDT2_thetahat_4_b,
        EDT2_x_6c_squared=EDT2_x_6c_squared,
        EDT2_x_6c=EDT2_x_6c,
        EDT2_phi_6_d=EDT2_phi_6_d,
        EDT2_l_c6_B=EDT2_l_c6_B,
        EDT2_phi_6_c=EDT2_phi_6_c,
        EDT2_phi_6_B=EDT2_phi_6_B,
        EDT2_q6_B=EDT2_q6_B,
        EDT2_h5_B=EDT2_h5_B,
        EDT2_l_46_j_squared=EDT2_l_46_j_squared,
        EDT2_l_46_j=EDT2_l_46_j,
        EDT2_gamma_4=EDT2_gamma_4,
        EDT2_gamma_6=EDT2_gamma_6,
        EDT2_thetatilde_4=EDT2_thetatilde_4,
        EDT2_x_c6_squared=EDT2_x_c6_squared,
        EDT2_x_c6=EDT2_x_c6,
        EDT2_phi_4_b=EDT2_phi_4_b,
        EDT2_thetatilde_6=EDT2_thetatilde_6,
        EDT2_thetatilde_6_a=EDT2_thetatilde_6_a,
        EDT2_thetatilde_6_b=EDT2_thetatilde_6_b,
        EDT2_l_cc_squared=EDT2_l_cc_squared,
        EDT2_l_cc_C=EDT2_l_cc_C,
        EDT2_phi_4_d=EDT2_phi_4_d,
        EDT2_h6_C=EDT2_h6_C,
        EDT2_h5_C=EDT2_h5_C,
        EDT2_x_56_squared=EDT2_x_56_squared,
        EDT2_x_56=EDT2_x_56,
        EDT2_l_5c_D=EDT2_l_5c_D,
        EDT2_phi_56_a=EDT2_phi_56_a,
        EDT2_phi_56_b=EDT2_phi_56_b,
        EDT2_phi_56=EDT2_phi_56,
        EDT2_q5_D=EDT2_q5_D,
        EDT2_phi_7_D=EDT2_phi_7_D,
        EDT2_h6_D=EDT2_h6_D,
    )
