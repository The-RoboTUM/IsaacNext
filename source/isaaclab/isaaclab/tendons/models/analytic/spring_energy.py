from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SpringEnergyOutput:
    energy: torch.Tensor
    per_tendon_energy: dict[str, torch.Tensor]
    active_masks: dict[str, torch.Tensor]


class SpringEnergyModel:
    """Spring energy and slack handling for analytic tendon length deltas."""

    def __init__(self, tendon_data):
        self.tendon_data = tendon_data

    def energy_from_deltas(self, deltas: dict[str, torch.Tensor]) -> SpringEnergyOutput:
        td = self.tendon_data
        stiffness = {
            "gst": td.gst_stiffness,
            "dft": td.dft_stiffness,
            "kft": td.kft_stiffness,
            "edt1": td.edt1_stiffness,
            "edt2": td.edt2_stiffness,
        }

        per_tendon_energy: dict[str, torch.Tensor] = {}
        active_masks: dict[str, torch.Tensor] = {}
        total_energy: torch.Tensor | None = None

        for name, delta_l in deltas.items():
            active = delta_l <= 0.0
            energy = 0.5 * stiffness[name] * delta_l**2
            active_energy = energy[active].sum()
            per_tendon_energy[name] = energy
            active_masks[name] = active
            total_energy = active_energy if total_energy is None else total_energy + active_energy

        if total_energy is None:
            first = next(iter(deltas.values()))
            total_energy = first.sum() * 0.0

        return SpringEnergyOutput(
            energy=total_energy,
            per_tendon_energy=per_tendon_energy,
            active_masks=active_masks,
        )
