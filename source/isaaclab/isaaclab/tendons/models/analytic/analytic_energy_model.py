from __future__ import annotations

import torch

from isaaclab.tendons.models.analytic.geometry.lengths import compute_all_tendon_delta_lengths
from isaaclab.tendons.models.base import TendonEnergyModel, TendonModelOutput
from isaaclab.tendons.models.analytic.spring_energy import SpringEnergyModel
from isaaclab.tendons.models.analytic.jit_model import (
    compute_analytic_tendon_energy_jit,
    compute_delta_l_s_jit,
)


class AnalyticTendonEnergyModel(TendonEnergyModel):
    """Analytic tendon model equivalent to the old monolithic tendon manager.

    Two execution paths are intentionally kept separate:
      - ``energy(..., debug=True)`` / ``energy_debug``: eager Python path with
        dictionaries and rich intermediate values.
      - ``energy_jit`` / ``delta_lengths_jit``: TorchScript path that returns
        tensors only and is suitable for per-step simulation.
    """

    def __init__(self, tendon_data):
        self.tendon_data = tendon_data
        self.tendon_data_jit = tendon_data.to_jit() if hasattr(tendon_data, "to_jit") else tendon_data
        self.spring_energy = SpringEnergyModel(tendon_data)

    def delta_lengths(self, joint_angles: torch.Tensor, *, debug: bool = False):
        return compute_all_tendon_delta_lengths(joint_angles, self.tendon_data, debug=debug)

    def delta_lengths_jit(self, joint_angles: torch.Tensor):
        """TorchScript delta-length forward pass.

        Returns:
            Tuple of ``(gst, dft, kft, edt1, edt2)`` delta-length tensors.
        """
        return compute_delta_l_s_jit(joint_angles, self.tendon_data_jit)

    def energy_jit(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """TorchScript energy forward pass used by ``TendonManager.compute_torques_jit``."""
        total_energy, _, _, _, _, _ = compute_analytic_tendon_energy_jit(joint_angles, self.tendon_data_jit)
        return total_energy

    def energy_debug(self, joint_angles: torch.Tensor) -> TendonModelOutput:
        """Eager debug path with all available intermediate tensors."""
        return self.energy(joint_angles, debug=True)

    def energy(self, joint_angles: torch.Tensor, *, debug: bool = False) -> TendonModelOutput:
        # Keep the original eager model as the source of debug information.
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


def compute_delta_l_s_jitted(joint_angles: torch.Tensor, tendon_data):
    """Compatibility helper for callers that need the scripted delta-length path."""
    tendon_data_jit = tendon_data.to_jit() if hasattr(tendon_data, "to_jit") else tendon_data
    return compute_delta_l_s_jit(joint_angles, tendon_data_jit)
