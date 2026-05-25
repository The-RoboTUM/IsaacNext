from __future__ import annotations

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.tendons.models.analytic.constants import dummy_randomization
from isaaclab.tendons.models.analytic.analytic_energy_model import AnalyticTendonEnergyModel
from isaaclab.tendons.models.base import TendonEnergyModel
from isaaclab.tendons.robot_io import TendonRobotIO
from isaaclab.tendons.models.analytic.tendon_data import TendonData
from isaaclab.tendons.torque_mapper import TendonTorqueMapper


class TendonManager:
    """Isaac integration layer for tendon models.

    The manager is intentionally model-agnostic: it can use the analytic model today
    and a neural-network energy model later, as long as the model implements
    `TendonEnergyModel.energy()`.

    The public methods now make the execution mode explicit:
      - ``compute_torques_debug`` / ``apply_debug``: eager path with debug info.
      - ``compute_torques_jit`` / ``apply_jit``: TorchScript analytic path.
      - ``compute_torques`` / ``apply``: preserves the previous eager behaviour.
    """

    def __init__(
        self,
        robot: Articulation,
        tendon_data: TendonData | None = None,
        *,
        model: TendonEnergyModel | None = None,
        robot_io: TendonRobotIO | None = None,
        torque_mapper: TendonTorqueMapper | None = None,
    ):
        self.robot = robot
        self.device = robot.device
        self.tendon_data = tendon_data if tendon_data is not None else TendonData(robot.num_instances, dummy_randomization)
        self.model = model if model is not None else AnalyticTendonEnergyModel(self.tendon_data)
        self.robot_io = robot_io if robot_io is not None else TendonRobotIO(robot)
        self.torque_mapper = torque_mapper if torque_mapper is not None else TendonTorqueMapper(self.device)

        # Compatibility attributes for older code that accessed these on TendonManager.
        self.link_indices_left_right = self.robot_io.link_indices_left_right
        self.joint_indices_left = self.robot_io.joint_indices_left
        self.joint_indices_right = self.robot_io.joint_indices_right
        self.hip_joint_indices = self.robot_io.hip_joint_indices
        self.hip_static_joint_indices = self.robot_io.hip_static_joint_indices
        self.foot_link_indices = self.robot_io.foot_link_indices

    def _compute_torques_from_energy(self, joint_angles: torch.Tensor, energy: torch.Tensor):
        if not torch.is_grad_enabled():
            raise RuntimeError("Cannot compute tendon torques: torch grad mode is disabled.")

        if not joint_angles.requires_grad:
            raise RuntimeError("Cannot compute tendon torques: joint_angles.requires_grad is False.")

        if not energy.requires_grad:
            raise RuntimeError(
                "Cannot compute tendon torques: energy.requires_grad is False. "
                "Use the eager energy path, not energy_jit(), when computing torques by autograd."
            )

        tendon_torques = torch.autograd.grad(
            outputs=energy,
            inputs=joint_angles,
            create_graph=False,
            allow_unused=False,
        )[0]

        batch_size = self.robot.num_instances
        return tendon_torques[:batch_size], tendon_torques[batch_size:]

    def compute_torques(self, *, debug: bool = False):
        """Differentiable eager torque computation.

        This path must stay eager because torques are computed as dE/dq using autograd.
        """
        with torch.inference_mode(False), torch.enable_grad():
            joint_angles = self.robot_io.get_leg_joint_angles(requires_grad=True)
            joint_angles = joint_angles.detach().clone().requires_grad_(True)

            output = self.model.energy(joint_angles, debug=debug)
            left, right = self._compute_torques_from_energy(joint_angles, output.energy)

        if debug:
            info = output.debug or {}
            info["tendon_torques_left"] = left
            info["tendon_torques_right"] = right
            return left, right, info

        return left, right

    def compute_torques_debug(self):
        return self.compute_torques(debug=True)

    def compute_torques_jit(self):
        """TorchScript-backed torque computation.

        The scripted part is the analytic energy/delta-length forward pass.  The
        final gradient is intentionally left in eager PyTorch so autograd can
        differentiate through the scripted graph.
        """
        with torch.inference_mode(False), torch.enable_grad():
            joint_angles = self.robot_io.get_leg_joint_angles(requires_grad=True)
            joint_angles = joint_angles.detach().clone().requires_grad_(True)
            if hasattr(self.model, "energy_jit"):
                energy = self.model.energy_jit(joint_angles)
            else:
                # Safe fallback for future non-analytic models that have not exposed
                # a scripted forward path yet.
                energy = self.model.energy(joint_angles, debug=False).energy

            return self._compute_torques_from_energy(joint_angles, energy)

    def apply(self, *, debug: bool = False, virtual_ground_height: float | None = None):
        """Eager apply path preserving the previous public behaviour."""
        batch_size = self.robot.num_instances
        if debug:
            left, right, info = self.compute_torques(debug=True)
        else:
            left, right = self.compute_torques(debug=False)
            info = None

        link_torques = self.torque_mapper.joint_to_link_torques(left, right, batch_size=batch_size)
        forces = self.robot_io.compute_external_forces(virtual_ground_height=virtual_ground_height)

        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=link_torques,
            body_ids=self.link_indices_left_right,
        )
        return info

    def apply_debug(self, virtual_ground_height: float | None = None):
        return self.apply(debug=True, virtual_ground_height=virtual_ground_height)

    def apply_jit(self, *, virtual_ground_height: float | None = None):
        """Apply torques using the JIT torque path.

        External force construction remains in ``TendonRobotIO`` because it uses
        Isaac Lab state and quaternion utilities.
        """
        batch_size = self.robot.num_instances
        left, right = self.compute_torques_jit()
        link_torques = self.torque_mapper.joint_to_link_torques_jit(left, right, batch_size=batch_size)
        forces = self.robot_io.compute_external_forces(virtual_ground_height=virtual_ground_height)

        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=link_torques,
            body_ids=self.link_indices_left_right,
        )
        return None
