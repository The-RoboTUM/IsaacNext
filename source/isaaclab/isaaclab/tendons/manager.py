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
        self.tendon_data = tendon_data if tendon_data is not None else TendonData(1, dummy_randomization)
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

    def compute_torques(self, *, debug: bool = False):
        batch_size = self.robot.num_instances
        joint_angles = self.robot_io.get_leg_joint_angles(requires_grad=True)

        output = self.model.energy(joint_angles, debug=debug)
        tendon_torques = torch.autograd.grad(
            outputs=output.energy,
            inputs=joint_angles,
            create_graph=False,
            allow_unused=False,
        )[0]

        left = tendon_torques[:batch_size]
        right = tendon_torques[batch_size:]

        if debug:
            info = output.debug or {}
            info["tendon_torques_left"] = left
            info["tendon_torques_right"] = right
            return left, right, info
        return left, right

    def compute_torques_debug(self):
        return self.compute_torques(debug=True)

    def compute_torques_jit(self):
        # Kept as a compatibility alias. The refactored implementation is eager
        # PyTorch to allow the model interface to be swapped for a NN model.
        return self.compute_torques(debug=False)

    def apply(self, *, debug: bool = False, virtual_ground_height: float | None = None):
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

    def apply_jit(self):
        return self.apply(debug=False)
