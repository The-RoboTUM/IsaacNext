from __future__ import annotations

import torch

from isaaclab.tendons.models.analytic.geometry.lengths import compute_all_tendon_delta_lengths
from isaaclab.tendons.models.base import TendonEnergyModel, TendonModelOutput
from isaaclab.tendons.models.analytic.spring_energy import SpringEnergyModel


class AnalyticTendonEnergyModel(TendonEnergyModel):
    """Analytic tendon model equivalent to the old monolithic tendon manager."""

    def __init__(self, tendon_data):
        self.tendon_data = tendon_data
        self.spring_energy = SpringEnergyModel(tendon_data)

    def delta_lengths(self, joint_angles: torch.Tensor, *, debug: bool = False):
        return compute_all_tendon_delta_lengths(joint_angles, self.tendon_data, debug=debug)

    def energy(self, joint_angles: torch.Tensor, *, debug: bool = False) -> TendonModelOutput:
        delta_output = self.delta_lengths(joint_angles, debug=debug)
        deltas = delta_output.as_dict()
        spring = self.spring_energy.energy_from_deltas(deltas)

        debug_info = None
        if debug:
            debug_info = {
                **(delta_output.debug or {}),

                "GST_not_slack": spring.active_masks["gst"],
                "DFT_not_slack": spring.active_masks["dft"],
                "KFT_not_slack": spring.active_masks["kft"],
                "EDT1_not_slack": spring.active_masks["edt1"],
                "EDT2_not_slack": spring.active_masks["edt2"],

                "GST_energy": spring.per_tendon_energy["gst"],
                "DFT_energy": spring.per_tendon_energy["dft"],
                "KFT_energy": spring.per_tendon_energy["kft"],
                "EDT1_energy": spring.per_tendon_energy["edt1"],
                "EDT2_energy": spring.per_tendon_energy["edt2"],
            }

        return TendonModelOutput(
            energy=spring.energy,
            per_tendon_energy=spring.per_tendon_energy,
            delta_lengths=deltas,
            debug=debug_info,
        )


def compute_delta_l_s(joint_angles: torch.Tensor, tendon_data):
    """Compatibility helper matching the old delta-length tuple return."""
    return compute_all_tendon_delta_lengths(joint_angles, tendon_data, debug=False).as_tuple()
