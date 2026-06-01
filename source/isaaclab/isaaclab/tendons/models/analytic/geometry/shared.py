from __future__ import annotations

from typing import NamedTuple

import torch

import isaaclab.tendons.models.analytic.indices as tids
from isaaclab.tendons.models.analytic.geometry.common import angle_from_sws
from isaaclab.tendons.models.analytic.geometry.kinematics import TendonCoordinates
from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


class SharedTendonGeometry(NamedTuple):
    """Intermediate geometric terms computed once and reused by tendon calculators.

    This NamedTuple is the single geometry payload used by both the eager debug
    wrappers and the TorchScript path. Field names preserve the original
    variable names to keep validation against old debug logs straightforward.
    """

    # GST shared terms
    GST_x_4prime6_squared: torch.Tensor
    GST_x_4prime6: torch.Tensor
    GST_l_4prime6_squared: torch.Tensor
    GST_l_4prime6: torch.Tensor
    GST_phi_4prime_a: torch.Tensor
    GST_phi_4prime_b: torch.Tensor
    GST_phi_4prime_B: torch.Tensor
    GST_h5_B: torch.Tensor
    GST_theta_6_a: torch.Tensor
    GST_theta_6_b: torch.Tensor
    GST_x_4prime7_squared: torch.Tensor
    GST_x_4prime7: torch.Tensor
    GST_phi_4prime_d: torch.Tensor
    GST_phi_4prime_c: torch.Tensor
    GST_phi_4prime_C: torch.Tensor
    GST_h5_C: torch.Tensor
    GST_h6_C: torch.Tensor
    GST_x_57_squared: torch.Tensor
    GST_x_57: torch.Tensor
    GST_l_57_squared: torch.Tensor
    GST_l_57: torch.Tensor
    GST_phi_5_a: torch.Tensor
    GST_phi_5_b: torch.Tensor
    GST_phi_5_D: torch.Tensor
    GST_h6_D: torch.Tensor
    # KFT shared terms
    KFT_l_8c_j_squared: torch.Tensor
    KFT_l_8c: torch.Tensor
    KFT_phi_8: torch.Tensor
    KFT_phi_8_a: torch.Tensor
    KFT_q8: torch.Tensor
    # DFT shared terms
    DFT_phi_4_a: torch.Tensor
    DFT_x_c6_squared: torch.Tensor
    DFT_x_c6: torch.Tensor
    DFT_l_c6_squared: torch.Tensor
    DFT_l_c6: torch.Tensor
    DFT_phi_4_b: torch.Tensor
    DFT_phi_4_B: torch.Tensor
    DFT_phi_6_B: torch.Tensor
    DFT_q_6_B: torch.Tensor
    DFT_h5_B: torch.Tensor
    DFT_theta_6_a: torch.Tensor
    DFT_theta_6_b: torch.Tensor
    DFT_l_c7_squared: torch.Tensor
    DFT_phi_4_d: torch.Tensor
    DFT_phi_4_C: torch.Tensor
    DFT_h5_C: torch.Tensor
    DFT_h6_C: torch.Tensor
    DFT_x_57_squared: torch.Tensor
    DFT_x_57: torch.Tensor
    DFT_l_57_squared: torch.Tensor
    DFT_l_57: torch.Tensor
    DFT_phi_5_a: torch.Tensor
    DFT_phi_5_b: torch.Tensor
    DFT_phi_5_D: torch.Tensor
    DFT_h6_D: torch.Tensor
    # EDT1 shared terms
    EDT1_x_c5_squared: torch.Tensor
    EDT1_x_c5: torch.Tensor
    EDT1_phi_4_a: torch.Tensor
    EDT1_thetahat_5_a: torch.Tensor
    EDT1_l_c5_A: torch.Tensor
    EDT1_phi_45_A: torch.Tensor
    EDT1_q5_A: torch.Tensor
    EDT1_thetahat_5_b: torch.Tensor
    EDT1_phi_4_b: torch.Tensor
    EDT1_h5_B: torch.Tensor
    EDT1_l_cc: torch.Tensor
    # EDT2 shared terms
    EDT2_x_c5_squared: torch.Tensor
    EDT2_x_c5: torch.Tensor
    EDT2_phi_4_a: torch.Tensor
    EDT2_thetahat_5_a: torch.Tensor
    EDT2_l_c5_A: torch.Tensor
    EDT2_phi_45_A: torch.Tensor
    EDT2_q5_A: torch.Tensor
    EDT2_x_64prime_squared: torch.Tensor
    EDT2_x_64prime: torch.Tensor
    EDT2_phi_6_a: torch.Tensor
    EDT2_thetahat_4_a: torch.Tensor
    EDT2_thetahat_4_b: torch.Tensor
    EDT2_x_6c_squared: torch.Tensor
    EDT2_x_6c: torch.Tensor
    EDT2_phi_6_d: torch.Tensor
    EDT2_l_c6_B: torch.Tensor
    EDT2_phi_6_c: torch.Tensor
    EDT2_phi_6_B: torch.Tensor
    EDT2_q6_B: torch.Tensor
    EDT2_h5_B: torch.Tensor
    EDT2_l_46_j_squared: torch.Tensor
    EDT2_l_46_j: torch.Tensor
    EDT2_gamma_4: torch.Tensor
    EDT2_gamma_6: torch.Tensor
    EDT2_thetatilde_4: torch.Tensor
    EDT2_x_c6_squared: torch.Tensor
    EDT2_x_c6: torch.Tensor
    EDT2_phi_4_b: torch.Tensor
    EDT2_thetatilde_6: torch.Tensor
    EDT2_thetatilde_6_a: torch.Tensor
    EDT2_thetatilde_6_b: torch.Tensor
    EDT2_l_cc_squared: torch.Tensor
    EDT2_l_cc_C: torch.Tensor
    EDT2_phi_4_d: torch.Tensor
    EDT2_h6_C: torch.Tensor
    EDT2_h5_C: torch.Tensor
    EDT2_x_56_squared: torch.Tensor
    EDT2_x_56: torch.Tensor
    EDT2_l_5c_D: torch.Tensor
    EDT2_phi_56_a: torch.Tensor
    EDT2_phi_56_b: torch.Tensor
    EDT2_phi_56: torch.Tensor
    EDT2_q5_D: torch.Tensor
    EDT2_phi_7_D: torch.Tensor
    EDT2_h6_D: torch.Tensor


def shared_geometry_as_debug_dict(
        geom: SharedTendonGeometry,
) -> dict[str, torch.Tensor]:
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


@torch.jit.script
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
            * torch.cos(thetas[:, tids.I_THETA_GST_5])
    )
    GST_x_4prime6 = torch.sqrt(GST_x_4prime6_squared)
    GST_l_4prime6_squared = (
            GST_x_4prime6_squared
            - (tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_6])
            ** 2
    )
    GST_l_4prime6 = torch.sqrt(GST_l_4prime6_squared)
    GST_phi_4prime_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.I_THETA_GST_5],
    )
    GST_phi_4prime_b = torch.acos(
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
    ] * torch.cos(GST_phi_4prime_B)

    GST_theta_6_a = torch.pi - thetas[:, tids.I_THETA_GST_5] - GST_phi_4prime_a
    GST_theta_6_b = thetas[:, tids.I_THETA_ALL_6] - GST_theta_6_a
    GST_x_4prime7_squared = (
            GST_x_4prime6_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2 * GST_x_4prime6 * tendon_data.link_lengths[:, tids.I_LINK_67] * torch.cos(GST_theta_6_b)
    )
    GST_x_4prime7 = torch.sqrt(GST_x_4prime7_squared)
    GST_phi_4prime_d = angle_from_sws(GST_x_4prime6, tendon_data.link_lengths[:, tids.I_LINK_67], GST_theta_6_b)
    GST_phi_4prime_c = torch.acos(tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] / GST_x_4prime7)
    GST_phi_4prime_C = GST_phi_4prime_a + GST_phi_4prime_c + GST_phi_4prime_d
    GST_h5_C = tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] - tendon_data.link_lengths[
        :, tids.I_LINK_4prime5
    ] * torch.cos(GST_phi_4prime_C)
    GST_h6_C = tendon_data.pulley_radii[:, tids.I_RADIUS_GST_4prime] - GST_x_4prime6 * torch.cos(
        GST_phi_4prime_c + GST_phi_4prime_d
    )

    GST_x_57_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(thetas[:, tids.I_THETA_ALL_6])
    )
    GST_x_57 = torch.sqrt(GST_x_57_squared)
    GST_l_57_squared = GST_x_57_squared - tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] ** 2
    GST_l_57 = torch.sqrt(GST_l_57_squared)
    GST_phi_5_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_67],
        thetas[:, tids.I_THETA_ALL_6],
    )
    GST_phi_5_b = torch.acos(tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] / GST_x_57)
    GST_phi_5_D = GST_phi_5_a + GST_phi_5_b
    GST_h6_D = tendon_data.pulley_radii[:, tids.I_RADIUS_GST_5] - tendon_data.link_lengths[
        :, tids.I_LINK_56
    ] * torch.cos(GST_phi_5_D)

    # ---------------- KFT ----------------
    KFT_l_8c_j_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_38]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_KFT_3C]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_38]
            * tendon_data.link_lengths[:, tids.I_LINK_KFT_3C]
            * torch.cos(theta_hats[:, tids.I_THETA_KFT_3])
    )
    KFT_l_8c = torch.sqrt(KFT_l_8c_j_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_KFT_8])
    KFT_phi_8 = torch.atan2(KFT_l_8c, tendon_data.pulley_radii[:, tids.I_RADIUS_KFT_8])
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
            * torch.cos(thetas[:, tids.I_THETA_DFT_5])
    )
    DFT_x_c6 = torch.sqrt(DFT_x_c6_squared)

    # state B
    DFT_l_c6_squared = DFT_x_c6_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_DFT_6]
    DFT_l_c6 = torch.sqrt(DFT_l_c6_squared)
    DFT_phi_4_b = torch.atan2(tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_6], DFT_l_c6)
    DFT_phi_4_B = DFT_phi_4_a + DFT_phi_4_b
    DFT_phi_6_B = torch.pi * 1.5 - DFT_phi_4_B - thetas[:, tids.I_THETA_DFT_5]
    DFT_q_6_B = (
            thetas[:, tids.I_THETA_ALL_6]
            - DFT_phi_6_B
            - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_DFT_6C_J6]
    )
    DFT_h5_B = tendon_data.link_lengths[:, tids.I_LINK_DFT_C5] * torch.sin(DFT_phi_4_B)

    # state C
    DFT_theta_6_a = torch.pi - thetas[:, tids.I_THETA_DFT_5] - DFT_phi_4_a
    DFT_theta_6_b = thetas[:, tids.I_THETA_ALL_6] - DFT_theta_6_a
    DFT_l_c7_squared = (
            DFT_x_c6_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2 * DFT_x_c6 * tendon_data.link_lengths[:, tids.I_LINK_67] * torch.cos(DFT_theta_6_b)
    )
    DFT_phi_4_d = angle_from_sws(DFT_x_c6, tendon_data.link_lengths[:, tids.I_LINK_67], DFT_theta_6_b)
    DFT_phi_4_C = DFT_phi_4_a + DFT_phi_4_d
    DFT_h5_C = tendon_data.link_lengths[:, tids.I_LINK_DFT_C5] * torch.sin(DFT_phi_4_C)
    DFT_h6_C = DFT_x_c6 * torch.sin(DFT_phi_4_d)

    # state D
    DFT_x_57_squared = GST_x_57_squared
    DFT_x_57 = GST_x_57
    DFT_l_57_squared = DFT_x_57_squared - tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5] ** 2
    DFT_l_57 = torch.sqrt(DFT_l_57_squared)
    DFT_phi_5_a = GST_phi_5_a
    DFT_phi_5_b = torch.acos(tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5] / DFT_x_57)
    DFT_phi_5_D = DFT_phi_5_a + DFT_phi_5_b
    DFT_h6_D = tendon_data.pulley_radii[:, tids.I_RADIUS_DFT_5] - tendon_data.link_lengths[
        :, tids.I_LINK_56
    ] * torch.cos(DFT_phi_5_D)

    # ---------------- EDT1 ----------------
    EDT1_x_c5_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_C4]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4]
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * torch.cos(theta_hats[:, tids.I_THETA_EDT1_4])
    )
    EDT1_x_c5 = torch.sqrt(EDT1_x_c5_squared)
    EDT1_phi_4_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT1_C4],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT1_4],
    )
    EDT1_thetahat_5_a = torch.pi - theta_hats[:, tids.I_THETA_EDT1_4] - EDT1_phi_4_a
    EDT1_l_c5_A = torch.sqrt(EDT1_x_c5_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT1_5])
    EDT1_phi_45_A = torch.atan2(EDT1_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT1_5])
    EDT1_q5_A = (
            2 * torch.pi
            - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_EDT1_5C_J5]
            - thetas[:, tids.I_THETA_EDT1_5]
            - EDT1_thetahat_5_a
            - EDT1_phi_45_A
    )
    EDT1_thetahat_5_b = theta_hats[:, tids.I_THETA_EDT1_5] - EDT1_thetahat_5_a
    EDT1_phi_4_b = angle_from_sws(EDT1_x_c5, tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C], EDT1_thetahat_5_b)
    EDT1_h5_B = EDT1_x_c5 * torch.sin(EDT1_phi_4_b)
    EDT1_l_cc = torch.sqrt(
        EDT1_x_c5_squared
        + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT1_5C]
        - 2 * EDT1_x_c5 * tendon_data.link_lengths[:, tids.I_LINK_EDT1_5C] * torch.cos(EDT1_thetahat_5_b)
    )

    ### ------------- EDT2 ------------- ###
    EDT2_x_c5_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4]
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * torch.cos(theta_hats[:, tids.I_THETA_EDT2_4])
    )
    EDT2_x_c5 = torch.sqrt(EDT2_x_c5_squared)
    EDT2_phi_4_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT2_4],
    )
    EDT2_thetahat_5_a = torch.pi - theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_phi_4_a

    # state A: tendon wraps around j5 and j6 pulleys
    EDT2_l_c5_A = torch.sqrt(EDT2_x_c5_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5])
    EDT2_phi_45_A = torch.atan2(EDT2_l_c5_A, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5])
    EDT2_q5_A = (
            2 * torch.pi
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
            * torch.cos(theta_hats[:, tids.I_THETA_EDT2_5])
    )
    EDT2_x_64prime = torch.sqrt(EDT2_x_64prime_squared)
    EDT2_phi_6_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        theta_hats[:, tids.I_THETA_EDT2_5],
    )
    EDT2_thetahat_4_a = torch.pi - theta_hats[:, tids.I_THETA_EDT2_5] - EDT2_phi_6_a
    EDT2_thetahat_4_b = theta_hats[:, tids.I_THETA_EDT2_4] - EDT2_thetahat_4_a
    EDT2_x_6c_squared = (
            EDT2_x_64prime_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
            - 2 * EDT2_x_64prime * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4] * torch.cos(EDT2_thetahat_4_b)
    )
    EDT2_x_6c = torch.sqrt(EDT2_x_6c_squared)
    EDT2_phi_6_d = angle_from_sws(
        EDT2_x_64prime,
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        EDT2_thetahat_4_b,
    )
    EDT2_l_c6_B = torch.sqrt(EDT2_x_6c_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_6])
    EDT2_phi_6_c = torch.atan2(EDT2_l_c6_B, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6])
    EDT2_phi_6_B = EDT2_phi_6_a + EDT2_phi_6_c + EDT2_phi_6_d
    EDT2_q6_B = (
            theta_hats[:, tids.I_THETA_ALL_6]
            - EDT2_phi_6_B
            - tendon_data.tendon_tangency_angles[:, tids.I_TENDON_TANGENCY_ANGLE_EDT2_67_J6]
    )
    EDT2_h5_B = tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_6] - tendon_data.link_lengths[
        :, tids.I_LINK_56
    ] * torch.cos(EDT2_phi_6_B)

    # state C: tendon does not wrap around any pulley
    EDT2_l_46_j_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * torch.cos(thetas[:, tids.I_THETA_EDT2_5])
    )
    EDT2_l_46_j = torch.sqrt(EDT2_l_46_j_squared)
    EDT2_gamma_4 = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_4prime5],
        tendon_data.link_lengths[:, tids.I_LINK_56],
        thetas[:, tids.I_THETA_EDT2_5],
    )
    EDT2_gamma_6 = torch.pi - EDT2_gamma_4 - thetas[:, tids.I_THETA_EDT2_5]
    EDT2_thetatilde_4 = theta_hats[:, tids.I_THETA_EDT2_4] + EDT2_gamma_4
    EDT2_x_c6_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_EDT2_C4]
            + EDT2_l_46_j_squared
            - 2 * tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4] * EDT2_l_46_j * torch.cos(EDT2_thetatilde_4)
    )
    EDT2_x_c6 = torch.sqrt(EDT2_x_c6_squared)
    EDT2_phi_4_b = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_EDT2_C4],
        EDT2_l_46_j,
        EDT2_thetatilde_4,
    )
    EDT2_thetatilde_6 = theta_hats[:, tids.I_THETA_ALL_6] + EDT2_gamma_6
    EDT2_thetatilde_6_a = torch.pi - EDT2_thetatilde_4 - EDT2_phi_4_b
    EDT2_thetatilde_6_b = EDT2_thetatilde_6 - EDT2_thetatilde_6_a
    EDT2_l_cc_squared = (
            EDT2_x_c6_squared
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2 * EDT2_x_c6 * tendon_data.link_lengths[:, tids.I_LINK_67] * torch.cos(EDT2_thetatilde_6_b)
    )
    EDT2_l_cc_C = torch.sqrt(EDT2_l_cc_squared)
    EDT2_phi_4_d = angle_from_sws(EDT2_x_c6, tendon_data.link_lengths[:, tids.I_LINK_67], EDT2_thetatilde_6_b)

    EDT2_h6_C = EDT2_x_c6 * torch.sin(EDT2_phi_4_d)
    EDT2_h5_C = EDT2_h6_C - tendon_data.link_lengths[:, tids.I_LINK_56] * torch.sin(EDT2_gamma_6)

    # state D: tendon wraps around j5 pulley but not j6 pulley
    EDT2_x_56_squared = (
            tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            + tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * tendon_data.link_lengths[:, tids.I_LINK_56]
            * tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(theta_hats[:, tids.I_THETA_ALL_6])
    )
    EDT2_x_56 = torch.sqrt(EDT2_x_56_squared)
    EDT2_l_5c_D = torch.sqrt(EDT2_x_56_squared - tendon_data.pulley_radii_squared[:, tids.I_RADIUS_EDT2_5])
    EDT2_phi_56_a = angle_from_sws(
        tendon_data.link_lengths[:, tids.I_LINK_56],
        tendon_data.link_lengths[:, tids.I_LINK_67],
        theta_hats[:, tids.I_THETA_ALL_6],
    )
    EDT2_phi_56_b = torch.atan2(EDT2_l_5c_D, tendon_data.pulley_radii[:, tids.I_RADIUS_EDT2_5])
    EDT2_phi_56 = EDT2_phi_56_a + EDT2_phi_56_b
    EDT2_q5_D = (
            2 * torch.pi
            - EDT2_phi_56
            - thetas[:, tids.I_THETA_EDT2_5]
            - EDT2_thetahat_5_a
            - EDT2_phi_45_A  # note: phi_45 is the same for states A and D
    )
    EDT2_phi_7_D = 1.5 * torch.pi - theta_hats[:, tids.I_THETA_ALL_6] - EDT2_phi_56
    EDT2_h6_D = tendon_data.link_lengths[:, tids.I_LINK_67] * torch.sin(EDT2_phi_7_D)

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
