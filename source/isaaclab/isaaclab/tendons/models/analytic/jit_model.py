# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TorchScript entry points for the analytic tendon model.

The math lives in ``geometry/`` and ``spring_energy.py``. This module is now
only a compatibility layer that exposes the same public JIT function names used
by the manager/model code.
"""

import torch

from isaaclab.tendons.models.analytic.geometry.lengths import compute_all_tendon_delta_lengths_jit
from isaaclab.tendons.models.analytic.spring_energy import compute_spring_energy_from_delta_lengths
from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


@torch.jit.script
def compute_delta_l_s_jit(
    joint_angles: torch.Tensor,
    tendon_data: TendonDataJIT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility alias for the scripted shared-geometry delta-length path."""
    return compute_all_tendon_delta_lengths_jit(joint_angles, tendon_data)


@torch.jit.script
def tendon_energy_from_delta_lengths_jit(
    GST_delta_L_s: torch.Tensor,
    DFT_delta_L_s: torch.Tensor,
    KFT_delta_L_s: torch.Tensor,
    EDT1_delta_L_s: torch.Tensor,
    EDT2_delta_L_s: torch.Tensor,
    tendon_data: TendonDataJIT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility alias for the scripted shared spring-energy path."""
    energy = compute_spring_energy_from_delta_lengths(
        GST_delta_L_s,
        DFT_delta_L_s,
        KFT_delta_L_s,
        EDT1_delta_L_s,
        EDT2_delta_L_s,
        tendon_data,
    )
    return (
        energy.energy,
        energy.gst_energy,
        energy.dft_energy,
        energy.kft_energy,
        energy.edt1_energy,
        energy.edt2_energy,
    )


@torch.jit.script
def compute_analytic_tendon_energy_jit(
    joint_angles: torch.Tensor,
    tendon_data: TendonDataJIT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jitted analytic forward pass returning total and per-tendon energies."""
    (
        GST_delta_L_s,
        DFT_delta_L_s,
        KFT_delta_L_s,
        EDT1_delta_L_s,
        EDT2_delta_L_s,
    ) = compute_delta_l_s_jit(joint_angles, tendon_data)

    (
        total_energy,
        GST_energy,
        DFT_energy,
        KFT_energy,
        EDT1_energy,
        EDT2_energy,
    ) = tendon_energy_from_delta_lengths_jit(
        GST_delta_L_s,
        DFT_delta_L_s,
        KFT_delta_L_s,
        EDT1_delta_L_s,
        EDT2_delta_L_s,
        tendon_data,
    )

    return (
        total_energy,
        GST_energy,
        DFT_energy,
        KFT_energy,
        EDT1_energy,
        EDT2_energy,
    )
