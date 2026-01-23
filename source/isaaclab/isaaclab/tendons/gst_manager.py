import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.tendons.constants import (
    tids,
    TendonData,
    dummy_randomization,
    link_names,
    joint_names,
    N_LINKS,
)


# todo comment
def angle_from_sws(a, b, theta):
    x = a - b * torch.cos(theta)
    y = b * torch.sin(theta)
    return torch.atan2(y, x)


# Running pipeline:
# 0) transform joint angles to thetas and qs
# 1) evaluate conditions
# 2) compute energy with conditional function
# 3) differentiate w.r.t. joint angles
# 4) apply torques


class GSTTendonManager:
    # affects joints j3, j4, j5, j6 through links s23, s34, s45, s56, s67
    def __init__(
        self,
        robot: Articulation,
    ):
        self.robot = robot
        self.device = robot.device

        self.link_indices, _ = self.robot.find_bodies(link_names, preserve_order=True)
        self.joint_indices, _ = self.robot.find_joints(joint_names, preserve_order=True)
        # TODO: add explanation in params to discuss these indices definitions
        self.tendon_data = TendonData(1, dummy_randomization)

    def apply(self):
        print("Applying GST tendon model...")
        joint_angles = (
            self.robot.data.joint_pos[:, self.joint_indices]
            .clone()
            .requires_grad_(True)
        )  # q3, q4, q5, q6
        # 0) transform joint angles to thetas and qs
        joint_angles_signed = self.tendon_data.joint_directions * joint_angles
        thetas = joint_angles_signed + self.tendon_data.joint_offsets_theta
        qs = joint_angles_signed + self.tendon_data.joint_offsets_q

        # 1) evaluate conditions
        # 1a) compute h5^B
        x_4prime6_squared = (
            self.tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
            + self.tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            - 2
            * self.tendon_data.link_lengths[:, tids.I_LINK_4prime5]
            * self.tendon_data.link_lengths[:, tids.I_LINK_56]
            * torch.cos(thetas[:, tids.I_JOINT_5])
        )
        x_4prime6 = torch.sqrt(x_4prime6_squared)
        l_4prime6_squared = (
            x_4prime6_squared
            - (
                self.tendon_data.pulley_radii[:, tids.I_RADIUS_4prime]
                - self.tendon_data.pulley_radii[:, tids.I_RADIUS_6]
            )
            ** 2
        )
        l_4prime6 = torch.sqrt(l_4prime6_squared)
        phi_4prime_a = angle_from_sws(
            self.tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            self.tendon_data.link_lengths[:, tids.I_LINK_56],
            thetas[:, tids.I_JOINT_5],
        )

        phi_4prime_b = torch.acos(
            (
                self.tendon_data.pulley_radii_squared[:, tids.I_RADIUS_4prime]
                + x_4prime6_squared
                - self.tendon_data.pulley_radii_squared[:, tids.I_RADIUS_6]
                - l_4prime6_squared
            )
            / (2 * self.tendon_data.pulley_radii[:, tids.I_RADIUS_4prime] * x_4prime6)
        )
        phi_4prime_B = phi_4prime_a + phi_4prime_b
        h5_B = self.tendon_data.pulley_radii[
            :, tids.I_RADIUS_4prime
        ] - self.tendon_data.link_lengths[:, tids.I_LINK_4prime5] * torch.cos(
            phi_4prime_B
        )

        # 1b) compute h5^C and h6^C
        theta_6_a = torch.pi - thetas[:, tids.I_JOINT_5] - phi_4prime_a
        theta_6_b = thetas[:, tids.I_JOINT_6] - theta_6_a
        x_4prime7_squared = (
            x_4prime6_squared
            + self.tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * x_4prime6
            * self.tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(theta_6_b)
        )
        x_4prime7 = torch.sqrt(x_4prime7_squared)
        phi_4prime_d = angle_from_sws(
            x_4prime6,
            self.tendon_data.link_lengths[:, tids.I_LINK_67],
            theta_6_b,
        )

        phi_4prime_c = torch.acos(
            self.tendon_data.pulley_radii[:, tids.I_RADIUS_4prime] / x_4prime7
        )
        phi_4prime_C = phi_4prime_a + phi_4prime_c + phi_4prime_d
        h5_C = self.tendon_data.pulley_radii[
            :, tids.I_RADIUS_4prime
        ] - self.tendon_data.link_lengths[:, tids.I_LINK_4prime5] * torch.cos(
            phi_4prime_C
        )
        h6_C = self.tendon_data.pulley_radii[
            :, tids.I_RADIUS_4prime
        ] - x_4prime6 * torch.cos(phi_4prime_c + phi_4prime_d)

        # print("Theta 6:", thetas[:, self.JOINT_ANGLES_6])

        # 1c) compute h6^D
        x_57_squared = (
            self.tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            + self.tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            - 2
            * self.tendon_data.link_lengths[:, tids.I_LINK_56]
            * self.tendon_data.link_lengths[:, tids.I_LINK_67]
            * torch.cos(
                thetas[:, tids.I_JOINT_6],
            )
        )
        x_57 = torch.sqrt(x_57_squared)
        l_57_squared = (
            x_57_squared - self.tendon_data.pulley_radii[:, tids.I_RADIUS_5] ** 2
        )
        l_57 = torch.sqrt(l_57_squared)

        phi_5_a = angle_from_sws(
            self.tendon_data.link_lengths[:, tids.I_LINK_56],
            self.tendon_data.link_lengths[:, tids.I_LINK_67],
            thetas[:, tids.I_JOINT_6],
        )

        phi_5_b = torch.acos(self.tendon_data.pulley_radii[:, tids.I_RADIUS_5] / x_57)
        phi_5_D = phi_5_a + phi_5_b
        h6_D = self.tendon_data.pulley_radii[
            :, tids.I_RADIUS_5
        ] - self.tendon_data.link_lengths[:, tids.I_LINK_56] * torch.cos(phi_5_D)

        h5_B_disengaged = torch.where(
            h5_B > self.tendon_data.pulley_radii[:, tids.I_RADIUS_5],
            True,
            False,
        )
        h5_C_disengaged = torch.where(
            h5_C > self.tendon_data.pulley_radii[:, tids.I_RADIUS_5],
            True,
            False,
        )
        h6_C_disengaged = torch.where(
            h6_C > self.tendon_data.pulley_radii[:, tids.I_RADIUS_6],
            True,
            False,
        )

        h6_D_disengaged = torch.where(
            h6_D > self.tendon_data.pulley_radii[:, tids.I_RADIUS_6],
            True,
            False,
        )
        state_C = (h5_B_disengaged & h6_C_disengaged) | (
            h6_D_disengaged & h5_C_disengaged
        )
        state_B = ~state_C & h5_B_disengaged
        state_D = ~state_C & h6_D_disengaged
        state_A = ~(state_B | state_C | state_D)

        # 2) compute energy with conditional function for lower tendon state length
        lower_tendon_state_length_after_4prime = torch.zeros_like(joint_angles[:, 0])
        # state A
        lower_tendon_state_length_after_4prime[state_A] = (
            self.tendon_data.tendon_section_lengths[state_A, tids.I_LINK_4prime5]
            + qs[state_A, tids.I_JOINT_5]
            * self.tendon_data.pulley_radii[state_A, tids.I_RADIUS_5]
            + self.tendon_data.tendon_section_lengths[state_A, tids.I_LINK_56]
            + qs[state_A, tids.I_JOINT_6]
            * self.tendon_data.pulley_radii[state_A, tids.I_RADIUS_6]
            + self.tendon_data.tendon_section_lengths[state_A, tids.I_LINK_67]
        )

        # state B
        q6_B = (
            thetas[state_B, tids.I_JOINT_6]
            - self.tendon_data.tendon_tangency_angles[
                state_B, tids.I_TENDON_TANGENGY_ANGLES_67_j6
            ]
            - 2 * torch.pi
            + phi_4prime_B[state_B]
            + thetas[state_B, tids.I_JOINT_5]
        )

        lower_tendon_state_length_after_4prime[state_B] = (
            l_4prime6[state_B]
            + q6_B * self.tendon_data.pulley_radii[state_B, tids.I_RADIUS_6]
            + self.tendon_data.tendon_section_lengths[state_B, tids.I_LINK_67]
        )

        # state C
        l_4prime7_squared = (
            x_4prime7_squared[state_C]
            - self.tendon_data.pulley_radii[state_C, tids.I_RADIUS_4prime] ** 2
        )
        l_4prime7 = torch.sqrt(l_4prime7_squared)
        lower_tendon_state_length_after_4prime[state_C] = l_4prime7

        # state D
        q5_D = (
            thetas[state_D, tids.I_JOINT_5]
            - self.tendon_data.tendon_tangency_angles[
                state_D, tids.I_TENDON_TANGENGY_ANGLES_45_j5
            ]
            - phi_5_D[state_D]
        )
        lower_tendon_state_length_after_4prime[state_D] = (
            self.tendon_data.tendon_section_lengths[state_D, tids.I_LINK_4prime5]
            + q5_D * self.tendon_data.pulley_radii[state_D, tids.I_RADIUS_5]
            + l_57[state_D]
        )

        q4prime = (
            self.tendon_data.lower_tendon_length
            - lower_tendon_state_length_after_4prime
        ) / self.tendon_data.pulley_radii[:, tids.I_RADIUS_4prime]

        q4 = (
            thetas[:, tids.I_JOINT_4]
            - q4prime
            - self.tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENGY_ANGLES_34_j4
            ]
            - self.tendon_data.tendon_tangency_angles[
                :, tids.I_TENDON_TANGENGY_ANGLES_45_j4
            ]
        )

        delta_L_s = (
            self.tendon_data.upper_tendon_length
            - self.tendon_data.spring_rest_length
            - self.tendon_data.tendon_section_lengths[:, tids.I_LINK_23]
            - qs[:, tids.I_JOINT_3] * self.tendon_data.pulley_radii[:, tids.I_RADIUS_3]
            - self.tendon_data.tendon_section_lengths[:, tids.I_LINK_34]
            - q4 * self.tendon_data.pulley_radii[:, tids.I_RADIUS_4]
        )

        not_slack = delta_L_s <= 0.0

        energy = 0.5 * self.tendon_data.stiffness * delta_L_s[not_slack] ** 2
        # 3) differentiate w.r.t. joint angles
        tendon_torques = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=joint_angles,
            create_graph=False,
            allow_unused=True,
        )[0]
        # 4) apply torques: with axis [0 1 0], to each link
        tendon_torques_full = torch.zeros(
            (joint_angles.shape[0], N_LINKS, 3), device=self.device
        )
        tendon_torques_full[:, :4, 2] = -tendon_torques
        tendon_torques_full[:, 1:, 2] += tendon_torques

        self.robot.set_external_force_and_torque(
            forces=torch.zeros((joint_angles.shape[0], N_LINKS, 3), device=self.device),
            torques=tendon_torques_full,
            body_ids=self.link_indices,
        )

        print("Applied GST tendon model.")
        return (
            state_A,
            state_B,
            state_C,
            state_D,
            not_slack,
            delta_L_s[0].item(),
            thetas[0].detach().cpu().numpy().tolist(),
        )
