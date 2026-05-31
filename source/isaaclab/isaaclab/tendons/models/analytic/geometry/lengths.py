from __future__ import annotations

import torch

from isaaclab.tendons.models.analytic.geometry.common import TendonDeltaLengths
from isaaclab.tendons.models.analytic.geometry.dft import compute_dft_delta_l, compute_dft_delta_l_core
from isaaclab.tendons.models.analytic.geometry.edt1 import compute_edt1_delta_l, compute_edt1_delta_l_core
from isaaclab.tendons.models.analytic.geometry.edt2 import compute_edt2_delta_l, compute_edt2_delta_l_core
from isaaclab.tendons.models.analytic.geometry.gst import compute_gst_delta_l, compute_gst_delta_l_core
from isaaclab.tendons.models.analytic.geometry.kft import compute_kft_delta_l, compute_kft_delta_l_core
from isaaclab.tendons.models.analytic.geometry.kinematics import compute_tendon_coordinates
from isaaclab.tendons.models.analytic.geometry.shared import (
    compute_shared_tendon_geometry,
    shared_geometry_as_debug_dict,
)
from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


def _as_jit_data(tendon_data):
    return tendon_data.to_jit() if hasattr(tendon_data, "to_jit") else tendon_data


@torch.jit.script
def compute_all_tendon_delta_lengths_jit(
    joint_angles: torch.Tensor,
    tendon_data: TendonDataJIT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TorchScript-compatible delta-length calculation using the shared geometry modules."""
    coords = compute_tendon_coordinates(joint_angles, tendon_data)
    geom = compute_shared_tendon_geometry(coords, tendon_data)

    gst = compute_gst_delta_l_core(coords, geom, tendon_data)
    dft = compute_dft_delta_l_core(coords, geom, tendon_data)
    kft = compute_kft_delta_l_core(coords, geom, tendon_data)
    edt1 = compute_edt1_delta_l_core(coords, geom, tendon_data)
    edt2 = compute_edt2_delta_l_core(coords, geom, tendon_data)

    return gst.delta_l, dft.delta_l, kft.delta_l, edt1.delta_l, edt2.delta_l


def compute_all_tendon_delta_lengths(joint_angles, tendon_data, *, debug: bool = False) -> TendonDeltaLengths:
    """Compute all tendon spring-length deltas using one shared geometry pass.

    This eager/debug wrapper calls the same tensor-only core functions as the
    JIT path, then packages the tensors into dictionaries/dataclasses for logs.
    """
    tendon_data_jit = _as_jit_data(tendon_data)
    coords = compute_tendon_coordinates(joint_angles, tendon_data_jit)
    geom = compute_shared_tendon_geometry(coords, tendon_data_jit)

    gst = compute_gst_delta_l(coords, geom, tendon_data_jit, debug=debug)
    dft = compute_dft_delta_l(coords, geom, tendon_data_jit, debug=debug)
    kft = compute_kft_delta_l(coords, geom, tendon_data_jit, debug=debug)
    edt1 = compute_edt1_delta_l(coords, geom, tendon_data_jit, debug=debug)
    edt2 = compute_edt2_delta_l(coords, geom, tendon_data_jit, debug=debug)

    debug_info = None
    if debug:
        debug_info = {
            "thetas": coords.thetas,
            "qs": coords.qs,
            "qhats": coords.qhats,
            **shared_geometry_as_debug_dict(geom),
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
