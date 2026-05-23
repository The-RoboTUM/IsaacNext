from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from isaaclab.tendons.models.analytic.tendon_data import TendonDataJIT


class SpringEnergyTensors(NamedTuple):
    energy: torch.Tensor
    gst_energy: torch.Tensor
    dft_energy: torch.Tensor
    kft_energy: torch.Tensor
    edt1_energy: torch.Tensor
    edt2_energy: torch.Tensor
    gst_active: torch.Tensor
    dft_active: torch.Tensor
    kft_active: torch.Tensor
    edt1_active: torch.Tensor
    edt2_active: torch.Tensor


@dataclass
class SpringEnergyOutput:
    energy: torch.Tensor
    per_tendon_energy: dict[str, torch.Tensor]
    active_masks: dict[str, torch.Tensor]


@torch.jit.script
def compute_spring_energy_from_delta_lengths(
    gst_delta_l: torch.Tensor,
    dft_delta_l: torch.Tensor,
    kft_delta_l: torch.Tensor,
    edt1_delta_l: torch.Tensor,
    edt2_delta_l: torch.Tensor,
    tendon_data: TendonDataJIT,
) -> SpringEnergyTensors:
    """Slack-aware spring energy shared by debug and JIT paths."""
    gst_energy = 0.5 * tendon_data.gst_stiffness * gst_delta_l**2
    dft_energy = 0.5 * tendon_data.dft_stiffness * dft_delta_l**2
    kft_energy = 0.5 * tendon_data.kft_stiffness * kft_delta_l**2
    edt1_energy = 0.5 * tendon_data.edt1_stiffness * edt1_delta_l**2
    edt2_energy = 0.5 * tendon_data.edt2_stiffness * edt2_delta_l**2

    gst_active = gst_delta_l <= 0.0
    dft_active = dft_delta_l <= 0.0
    kft_active = kft_delta_l <= 0.0
    edt1_active = edt1_delta_l <= 0.0
    edt2_active = edt2_delta_l <= 0.0

    total_energy = (
        torch.where(gst_active, gst_energy, torch.zeros_like(gst_energy)).sum()
        + torch.where(dft_active, dft_energy, torch.zeros_like(dft_energy)).sum()
        + torch.where(kft_active, kft_energy, torch.zeros_like(kft_energy)).sum()
        + torch.where(edt1_active, edt1_energy, torch.zeros_like(edt1_energy)).sum()
        + torch.where(edt2_active, edt2_energy, torch.zeros_like(edt2_energy)).sum()
    )

    return SpringEnergyTensors(
        energy=total_energy,
        gst_energy=gst_energy,
        dft_energy=dft_energy,
        kft_energy=kft_energy,
        edt1_energy=edt1_energy,
        edt2_energy=edt2_energy,
        gst_active=gst_active,
        dft_active=dft_active,
        kft_active=kft_active,
        edt1_active=edt1_active,
        edt2_active=edt2_active,
    )


class SpringEnergyModel:
    """Spring energy and slack handling for analytic tendon length deltas."""

    def __init__(self, tendon_data):
        self.tendon_data = tendon_data
        self.tendon_data_jit = tendon_data.to_jit() if hasattr(tendon_data, "to_jit") else tendon_data

    def energy_from_delta_tuple(
        self,
        deltas: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> SpringEnergyOutput:
        tensors = compute_spring_energy_from_delta_lengths(
            deltas[0],
            deltas[1],
            deltas[2],
            deltas[3],
            deltas[4],
            self.tendon_data_jit,
        )
        return SpringEnergyOutput(
            energy=tensors.energy,
            per_tendon_energy={
                "gst": tensors.gst_energy,
                "dft": tensors.dft_energy,
                "kft": tensors.kft_energy,
                "edt1": tensors.edt1_energy,
                "edt2": tensors.edt2_energy,
            },
            active_masks={
                "gst": tensors.gst_active,
                "dft": tensors.dft_active,
                "kft": tensors.kft_active,
                "edt1": tensors.edt1_active,
                "edt2": tensors.edt2_active,
            },
        )

    def energy_from_deltas(self, deltas: dict[str, torch.Tensor]) -> SpringEnergyOutput:
        return self.energy_from_delta_tuple(
            (
                deltas["gst"],
                deltas["dft"],
                deltas["kft"],
                deltas["edt1"],
                deltas["edt2"],
            )
        )
