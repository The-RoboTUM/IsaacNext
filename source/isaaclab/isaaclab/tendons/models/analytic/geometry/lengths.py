from __future__ import annotations

from isaaclab.tendons.models.analytic.geometry.common import TendonDeltaLengths
from isaaclab.tendons.models.analytic.geometry.dft import compute_dft_delta_l
from isaaclab.tendons.models.analytic.geometry.edt1 import compute_edt1_delta_l
from isaaclab.tendons.models.analytic.geometry.edt2 import compute_edt2_delta_l
from isaaclab.tendons.models.analytic.geometry.gst import compute_gst_delta_l
from isaaclab.tendons.models.analytic.geometry.kft import compute_kft_delta_l
from isaaclab.tendons.models.analytic.geometry.kinematics import compute_tendon_coordinates
from isaaclab.tendons.models.analytic.geometry.shared import compute_shared_tendon_geometry


def compute_all_tendon_delta_lengths(joint_angles, tendon_data, *, debug: bool = False) -> TendonDeltaLengths:
    """Compute all tendon spring-length deltas using one shared geometry pass."""
    coords = compute_tendon_coordinates(joint_angles, tendon_data)
    geom = compute_shared_tendon_geometry(coords, tendon_data)

    gst = compute_gst_delta_l(coords, geom, tendon_data, debug=debug)
    dft = compute_dft_delta_l(coords, geom, tendon_data, debug=debug)
    kft = compute_kft_delta_l(coords, geom, tendon_data, debug=debug)
    edt1 = compute_edt1_delta_l(coords, geom, tendon_data, debug=debug)
    edt2 = compute_edt2_delta_l(coords, geom, tendon_data, debug=debug)

    debug_info = None
    if debug:
        debug_info = {
            "thetas": coords.thetas,
            "qs": coords.qs,
            "qhats": coords.qhats,
            **geom.as_debug_dict(),
            **(gst.debug or {}),
            **(dft.debug or {}),
            **(kft.debug or {}),
            **(edt1.debug or {}),
            **(edt2.debug or {}),
        }

    return TendonDeltaLengths(
        gst=gst.delta_l,
        dft=dft.delta_l,
        kft=kft.delta_l,
        edt1=edt1.delta_l,
        edt2=edt2.delta_l,
        debug=debug_info,
    )
