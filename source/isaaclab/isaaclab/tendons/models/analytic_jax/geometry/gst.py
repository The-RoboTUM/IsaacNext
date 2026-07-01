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


class GSTDeltaCoreOutput(NamedTuple):
    delta_l: jnp.ndarray
    GST_state_A: jnp.ndarray
    GST_state_B: jnp.ndarray
    GST_state_C: jnp.ndarray
    GST_state_D: jnp.ndarray
    GST_q4: jnp.ndarray
    GST_q4prime: jnp.ndarray
    GST_q5_D: jnp.ndarray
    GST_q6_B: jnp.ndarray
    GST_l_4prime7: jnp.ndarray
    GST_lower_tendon_state_length_after_4prime: jnp.ndarray


def compute_gst_delta_l_core(
    coords: TendonCoordinates,
    geom: SharedTendonGeometry,
    tendon_data: TendonDataJIT,
) -> GSTDeltaCoreOutput:
    """GST spring-length delta.

    This tensor-only function is the single source of GST length math. The eager
    debug wrapper below only packages its outputs into dictionaries.
    """
    thetas = coords.thetas
    qs = coords.qs

    GST_h5_B_disengaged = geom.GST_h5_B > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
    GST_h5_C_disengaged = geom.GST_h5_C > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
    GST_h6_C_disengaged = geom.GST_h6_C > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
    GST_h6_D_disengaged = geom.GST_h6_D > tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]

    GST_state_C = (GST_h5_B_disengaged & GST_h6_C_disengaged) | (GST_h6_D_disengaged & GST_h5_C_disengaged)
    GST_state_B = ~GST_state_C & GST_h5_B_disengaged
    GST_state_D = ~GST_state_C & GST_h6_D_disengaged
    GST_state_A = ~(GST_state_B | GST_state_C | GST_state_D)

    GST_lower_tendon_state_length_after_4prime_A = (
        tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5]
        + qs[:, tids.I_Q_GST_5] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_56]
        + qs[:, tids.I_Q_GST_6] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_67]
    )

    GST_q6_B = (
        thetas[:, tids.I_THETA_ALL_6]
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_GST_67_J6]
        - 2 * jnp.pi
        + geom.GST_phi_4prime_B
        + thetas[:, tids.I_THETA_GST_5]
    )
    GST_lower_tendon_state_length_after_4prime_B = (
        geom.GST_l_4prime6
        + GST_q6_B * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6]
        + tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_67]
    )

    GST_l_4prime7_squared = geom.GST_x_4prime7_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_GST_4prime]
    GST_l_4prime7 = jnp.sqrt(GST_l_4prime7_squared)
    GST_lower_tendon_state_length_after_4prime_C = GST_l_4prime7

    GST_q5_D = (
        thetas[:, tids.I_THETA_GST_5]
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J5]
        - geom.GST_phi_5_D
    )
    GST_lower_tendon_state_length_after_4prime_D = (
        tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5]
        + GST_q5_D * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5]
        + geom.GST_l_57
    )

    GST_lower_tendon_state_length_after_4prime = jnp.where(
        GST_state_A,
        GST_lower_tendon_state_length_after_4prime_A,
        jnp.where(
            GST_state_B,
            GST_lower_tendon_state_length_after_4prime_B,
            jnp.where(
                GST_state_C, GST_lower_tendon_state_length_after_4prime_C, GST_lower_tendon_state_length_after_4prime_D
            ),
        ),
    )

    GST_q4prime = (
        tendon_data.lower_gst_length - GST_lower_tendon_state_length_after_4prime
    ) / tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime]
    GST_q4_base = (
        thetas[:, tids.I_THETA_GST_4]
        - GST_q4prime
        - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_GST_34_J4]
    )
    GST_q4_adjustment = jnp.where(
        jnp.logical_or(GST_state_A, GST_state_D),
        tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J4],
        jnp.where(GST_state_B, geom.GST_phi_4prime_B, geom.GST_phi_4prime_C),
    )
    GST_q4 = GST_q4_base - GST_q4_adjustment

    GST_delta_L_s = (
        tendon_data.upper_gst_length
        - tendon_data.gst_spring_rest_length
        - tendon_data.link_lengths[:, tids.I_LINK_GST_23]
        - qs[:, tids.I_Q_GST_3] * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_3]
        - tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_34]
        - GST_q4 * tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4]
    )

    return GSTDeltaCoreOutput(
        delta_l=GST_delta_L_s,
        GST_state_A=GST_state_A,
        GST_state_B=GST_state_B,
        GST_state_C=GST_state_C,
        GST_state_D=GST_state_D,
        GST_q4=GST_q4,
        GST_q4prime=GST_q4prime,
        GST_q5_D=GST_q5_D,
        GST_q6_B=GST_q6_B,
        GST_l_4prime7=GST_l_4prime7,
        GST_lower_tendon_state_length_after_4prime=GST_lower_tendon_state_length_after_4prime,
    )


def compute_gst_delta_l(coords, geom, tendon_data, *, debug: bool = False) -> TendonLengthOutput:
    core = compute_gst_delta_l_core(coords, geom, tendon_data)
    state = {
        "GST_state_a": core.GST_state_A,
        "GST_state_b": core.GST_state_B,
        "GST_state_c": core.GST_state_C,
        "GST_state_d": core.GST_state_D,
    }
    debug_info = None
    if debug:
        debug_info = {
            **state,
            "GST_delta_L_s": core.delta_l,
            "GST_q4": core.GST_q4,
            "GST_q4prime": core.GST_q4prime,
            "GST_q5_D": core.GST_q5_D,
            "GST_q6_B": core.GST_q6_B,
            "GST_l_4prime7": core.GST_l_4prime7,
            "GST_lower_tendon_state_length_after_4prime": core.GST_lower_tendon_state_length_after_4prime,
        }

    return TendonLengthOutput(delta_l=core.delta_l, state=state, debug=debug_info)
