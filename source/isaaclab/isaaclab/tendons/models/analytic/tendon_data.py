"""Tendon data for parallel training."""

from isaaclab.tendons.models.analytic.utils import list_from_dict
import torch

from isaaclab.tendons.models.analytic.constants import (
    N_CHAIN_LINKS_PER_LEG,
    N_CONNECTOR_OFFSETS,
    N_Q_OFFSETS,
    N_QHAT_OFFSETS,
    N_TENDON_SECTION_LENGTHS,
    N_TENDON_THETA_OFFSETS,
    N_TENDON_TANGENCY_ANGLES,
    TendonConstantRandomizationRanges,
    TendonConstants,
    dev,
    N_JOINTS,
    N_RADII,
    dummy_randomization,
)

import isaaclab.tendons.models.analytic.indices as tids


def same_sided_wrap(
        r_a: torch.Tensor, r_b: torch.Tensor, l_ab_j: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute same-sided wrap angle around two pulleys.

    Args:
        r_a: Radius of pulley A. Shape (B,).
        r_b: Radius of pulley B. Shape (B,).
        d_ab: Distance between pulleys A and B. Shape (B,).
    Returns:
        Wrap angles in radians and tendon length between pulleys. Shape (B,) each.
    """
    sin_phi_0 = (r_b - r_a) / l_ab_j
    phi_0 = torch.asin(sin_phi_0)
    phi_ab_ja = torch.pi / 2 + phi_0
    phi_ab_jb = torch.pi / 2 - phi_0
    l_ab = l_ab_j * torch.cos(phi_0)
    return phi_ab_ja, phi_ab_jb, l_ab


def opposite_sided_wrap(
        r_a: torch.Tensor, r_b: torch.Tensor, l_ab_j: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute opposite-sided wrap angle around two pulleys.

    Args:
        r_a: Radius of pulley A. Shape (B,).
        r_b: Radius of pulley B. Shape (B,).
        d_ab: Distance between pulleys A and B. Shape (B,).
    Returns:
        Wrap angles in radians and tendon length between pulleys. Shape (B,) each.
    """
    sin_phi_0 = (r_a + r_b) / l_ab_j
    phi_0 = torch.asin(sin_phi_0)
    phi_ab_ja = phi_ab_jb = torch.pi / 2 - phi_0
    l_ab = l_ab_j * torch.cos(phi_0)
    return phi_ab_ja, phi_ab_jb, l_ab


@torch.jit.script
class TendonDataJIT:
    def __init__(
            self,
            gst_stiffness: torch.Tensor,
            gst_spring_rest_length: torch.Tensor,
            upper_gst_length: torch.Tensor,
            lower_gst_length: torch.Tensor,
            dft_stiffness: torch.Tensor,
            dft_length: torch.Tensor,
            edt1_stiffness: torch.Tensor,
            edt1_length: torch.Tensor,
            edt2_stiffness: torch.Tensor,
            edt2_length: torch.Tensor,
            kft_stiffness: torch.Tensor,
            kft_length: torch.Tensor,
            joint_directions: torch.Tensor,
            pulley_radii: torch.Tensor,
            pulley_radii_squared: torch.Tensor,
            link_lengths: torch.Tensor,
            link_lengths_squared: torch.Tensor,
            tendon_offsets_theta: torch.Tensor,
            tendon_offsets_q_theta: torch.Tensor,
            tendon_offsets_qhat_thetahat: torch.Tensor,
            tendon_section_lengths: torch.Tensor,
            tendon_tangency_angles: torch.Tensor,
    ) -> None:
        """Convert tensor inputs into a JIT-compatible TendonDataJIT."""
        self.gst_stiffness = gst_stiffness
        self.gst_spring_rest_length = gst_spring_rest_length
        self.upper_gst_length = upper_gst_length
        self.lower_gst_length = lower_gst_length
        self.dft_stiffness = dft_stiffness
        self.dft_length = dft_length
        self.edt1_stiffness = edt1_stiffness
        self.edt1_length = edt1_length
        self.edt2_stiffness = edt2_stiffness
        self.edt2_length = edt2_length
        self.kft_stiffness = kft_stiffness
        self.kft_length = kft_length
        self.joint_directions = joint_directions
        self.pulley_radii = pulley_radii
        self.pulley_radii_squared = pulley_radii_squared
        self.link_lengths = link_lengths
        self.link_lengths_squared = link_lengths_squared
        self.tendon_offsets_theta = tendon_offsets_theta
        self.tendon_offsets_q_theta = tendon_offsets_q_theta
        self.tendon_offsets_qhat_thetahat = tendon_offsets_qhat_thetahat
        self.tendon_section_lengths = tendon_section_lengths
        self.tendon_tangency_angles = tendon_tangency_angles


def as_tensor_on_device(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.detach().clone().to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


# we compute theta offsets (all tendon thetas), q offsets, tendon section lengths, tendon tangency angles,
# pulley radii,  link lengths (chain and connector)
class TendonData:
    """Tendon data for for parallel training.

    Includes randomization, derived constants, and batching.
    """

    def __init__(
            self,
            batch_size: int,
            randomization_ranges: TendonConstantRandomizationRanges,
            tc: TendonConstants | None = None,
    ) -> None:
        """Initialize tendon data."""
        batch_size *= 2  # for left and right legs
        tc = TendonConstants() if tc is None else tc

        joint_offsets_theta = torch.stack(
            [
                tc.joint_offsets_theta[i]
                + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.joint_offsets_theta[i])
                for i in range(N_JOINTS)
            ],
            dim=1,
        )  # (B, N_JOINTS)
        assert joint_offsets_theta.shape == (batch_size, N_JOINTS)

        pulley_radii = torch.stack(
            [
                as_tensor_on_device(tc.pulley_radii[i], dev)
                + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.pulley_radii[i])
                for i in range(N_RADII)
            ],
            dim=1,
        )  # (B, N_RADII)
        assert pulley_radii.shape == (batch_size, N_RADII)

        chain_link_lengths = torch.stack(
            [
                as_tensor_on_device(tc.chain_link_lengths[i], dev)
                + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.chain_link_lengths[i])
                for i in range(N_CHAIN_LINKS_PER_LEG)
            ],
            dim=1,
        )  # (B, N_CHAIN_LINKS)
        assert chain_link_lengths.shape == (batch_size, N_CHAIN_LINKS_PER_LEG)

        connector_link_lengths_longitudinal = torch.stack(
            [
                as_tensor_on_device(tc.connector_link_lengths_longitudinal[i], dev)
                + torch.empty(batch_size, device=dev).uniform_(
                    *randomization_ranges.connector_link_lengths_longitudinal[i]
                )
                for i in range(N_CONNECTOR_OFFSETS)
            ],
            dim=1,
        )  # (B, N_TENDON_ATTACHMENT_LINKS)
        assert connector_link_lengths_longitudinal.shape == (
            batch_size,
            N_CONNECTOR_OFFSETS,
        )

        connector_link_lengths_lateral = torch.stack(
            [
                as_tensor_on_device(tc.connector_link_lengths_lateral[i], dev)
                + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.connector_link_lengths_lateral[i])
                for i in range(N_CONNECTOR_OFFSETS)
            ],
            dim=1,
        )  # (B, N_CONNECTOR_OFFSETS)
        assert connector_link_lengths_lateral.shape == (batch_size, N_CONNECTOR_OFFSETS)

        # ----------------------connector link lengths------------------ #
        connector_link_lengths = torch.sqrt(
            connector_link_lengths_longitudinal ** 2 + connector_link_lengths_lateral ** 2
        )  # (B, N_CONNECTOR_OFFSETS)

        # ----------------------GST ------------------ #
        gst_stiffness = tc.gst_stiffness + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.gst_stiffness
        )
        gst_spring_rest_length = tc.gst_spring_rest_length + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.gst_spring_rest_length
        )

        gst_phi_34_j3, gst_phi_34_j4, gst_l_34 = opposite_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_GST_3],
            pulley_radii[:, tids.I_RADIUS_GST_4],
            chain_link_lengths[:, tids.I_CHAIN_LINK_34],
        )

        gst_phi_4prime5_j4, gst_phi_4prime5_j5, gst_l_4prime5 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_GST_4prime],
            pulley_radii[:, tids.I_RADIUS_GST_5],
            chain_link_lengths[:, tids.I_CHAIN_LINK_4prime5],
        )

        gst_phi_56_j5, gst_phi_56_j6, gst_l_56 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_GST_5],
            pulley_radii[:, tids.I_RADIUS_GST_6],
            chain_link_lengths[:, tids.I_CHAIN_LINK_56],
        )

        gst_phi_67_j6, _, gst_l_67 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_GST_6],
            torch.tensor(0.0, device=dev),
            chain_link_lengths[:, tids.I_CHAIN_LINK_67],
        )

        gst_l_2prime3 = connector_link_lengths_longitudinal[:, tids.I_CONNECTOR_LINK_GST_23]
        gst_phi_23_j3 = tc.gst_phi_23_j3 + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.gst_phi_23_j3
        )
        gst_angle_4prime5_to_j44prime = tc.angle_4prime5_to_j44prime + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.angle_4prime5_to_j44prime
        )

        gst_q_3_offset = -gst_phi_23_j3 - gst_phi_34_j3
        gst_q_4_offset = -gst_angle_4prime5_to_j44prime - gst_phi_34_j4

        gst_q_5_offset = -gst_phi_4prime5_j5 - gst_phi_56_j5
        gst_q_6_offset = -gst_phi_56_j6 - gst_phi_67_j6

        # Note: we randomize upper and lower tendon lengths after computing other offsets because of manufacturing tolerances.
        upper_gst_length = tc.upper_gst_length + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.upper_gst_length
        )
        lower_gst_length = tc.lower_gst_length + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.lower_gst_length
        )

        # -------------------- DFT ------------------ #
        dft_stiffness = tc.dft_stiffness + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.dft_stiffness
        )
        dft_length = tc.dft_length + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.dft_length)
        _, dft_phi_c5_j5, dft_l_c5 = same_sided_wrap(
            torch.tensor(0.0, device=dev),
            pulley_radii[:, tids.I_RADIUS_DFT_5],
            connector_link_lengths[:, tids.I_CONNECTOR_LINK_DFT_C5],
        )
        dft_phi_56_j5, dft_phi_56_j6, dft_l_56 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_DFT_5],
            pulley_radii[:, tids.I_RADIUS_DFT_6],
            chain_link_lengths[:, tids.I_CHAIN_LINK_56],
        )
        dft_phi_6c_j6, _, dft_l_6c = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_DFT_6],
            torch.tensor(0.0, device=dev),
            chain_link_lengths[:, tids.I_CHAIN_LINK_67],
        )

        dft_theta_offset_5 = joint_offsets_theta[
                                 :, tids.I_JOINT_5
                             ] - torch.atan2(
            # smaller theta because connector is on the side where link-theta is measured
            connector_link_lengths_lateral[:, tids.I_CONNECTOR_LINK_DFT_C5],
            connector_link_lengths_longitudinal[:, tids.I_CONNECTOR_LINK_DFT_C5],
        )

        dft_q5_offset = -dft_phi_c5_j5 - dft_phi_56_j5
        dft_q6_offset = -dft_phi_56_j6 - dft_phi_6c_j6

        # -------------------- EDT1 ----------------- #
        edt1_length = tc.edt1_length + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.edt1_length)
        edt1_stiffness = tc.edt1_stiffness + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.edt1_stiffness
        )

        edt1_theta_offset_4 = joint_offsets_theta[
                                  :, tids.I_JOINT_4
                              ] + torch.atan2(
            # larger theta because connector is opposite to the side where link-theta is measured
            connector_link_lengths_lateral[:, tids.I_CONNECTOR_LINK_EDT1_C4],
            connector_link_lengths_longitudinal[:, tids.I_CONNECTOR_LINK_EDT1_C4],
        )
        edt1_theta_offset_5 = joint_offsets_theta[
                                  :, tids.I_JOINT_5
                              ] + torch.atan2(
            # larger theta because connector is opposite to the side where link-theta is measured
            connector_link_lengths_lateral[:, tids.I_CONNECTOR_LINK_EDT1_5C],
            connector_link_lengths_longitudinal[:, tids.I_CONNECTOR_LINK_EDT1_5C],
        )

        edt1_phi_5c_jc, _, edt1_l_5c = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_EDT1_5],
            torch.tensor(0.0, device=dev),
            connector_link_lengths[:, tids.I_CONNECTOR_LINK_EDT1_5C],
        )

        # -------------------- EDT2 ----------------- #
        edt2_length = tc.edt2_length + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.edt2_length)
        edt2_stiffness = tc.edt2_stiffness + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.edt2_stiffness
        )

        edt2_phi_56_j5, edt2_phi_56_j6, edt2_l_56 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_EDT2_5],
            pulley_radii[:, tids.I_RADIUS_EDT2_6],
            chain_link_lengths[:, tids.I_CHAIN_LINK_56],
        )

        edt2_phi_6c_j6, _, edt2_l_6c = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_EDT2_6],
            torch.tensor(0.0, device=dev),
            chain_link_lengths[:, tids.I_CHAIN_LINK_67],
        )

        edt2_theta_offset_4 = joint_offsets_theta[
                                  :, tids.I_JOINT_4
                              ] + torch.atan2(
            # larger theta because connector is opposite to the side where link-theta is measured
            connector_link_lengths_lateral[:, tids.I_CONNECTOR_LINK_EDT2_C4],
            connector_link_lengths_longitudinal[:, tids.I_CONNECTOR_LINK_EDT2_C4],
        )

        edt2_q6hat_offset = -edt2_phi_56_j6 - edt2_phi_6c_j6

        # -------------------- KFT ----------------- #
        kft_length = tc.kft_length + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.kft_length)
        kft_stiffness = tc.kft_stiffness + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.kft_stiffness
        )

        kft_theta_offset_3 = joint_offsets_theta[
                                 :, tids.I_JOINT_3
                             ] - torch.atan2(
            # smaller theta because connector is on the same side as where link-theta is measured
            connector_link_lengths_lateral[:, tids.I_CONNECTOR_LINK_KFT_3C],
            connector_link_lengths_longitudinal[:, tids.I_CONNECTOR_LINK_KFT_3C],
        )

        # ----------------- Combined arrays ----------------- #
        # sec lengths -> gst: 2-3, 3-4, 4'-5, 5-6, 6-7; dft: c5, 5-6, 6-c; edt1: 5-c; edt2: 5-6, 6-c
        tendon_section_lengths = torch.stack(
            list_from_dict(
                {
                    tids.I_TENDON_SECTION_LENGTH_GST_23: gst_l_2prime3,
                    tids.I_TENDON_SECTION_LENGTH_GST_34: gst_l_34,
                    tids.I_TENDON_SECTION_LENGTH_GST_4PRIME5: gst_l_4prime5,
                    tids.I_TENDON_SECTION_LENGTH_GST_56: gst_l_56,
                    tids.I_TENDON_SECTION_LENGTH_GST_67: gst_l_67,
                    tids.I_TENDON_SECTION_LENGTH_DFT_C5: dft_l_c5,
                    tids.I_TENDON_SECTION_LENGTH_DFT_56: dft_l_56,
                    tids.I_TENDON_SECTION_LENGTH_DFT_6C: dft_l_6c,
                    tids.I_TENDON_SECTION_LENGTH_EDT1_5C: edt1_l_5c,
                    tids.I_TENDON_SECTION_LENGTH_EDT2_56: edt2_l_56,
                    tids.I_TENDON_SECTION_LENGTH_EDT2_6C: edt2_l_6c,
                },
                N_TENDON_SECTION_LENGTHS,
            ),
            dim=1,
        )  # (B, N_TENDON_SECTION_LENGTHS)

        # tangency angles -> kft: none, edt1: 5c-j5, edt2: 56-j5;
        # gst: 34-j4, 4'5-j4, 4'5-j5, 67-j6
        tendon_tangency_angles = torch.stack(
            list_from_dict(
                {
                    tids.I_TENDON_TANGENCY_ANGLE_GST_34_J4: gst_phi_34_j4,
                    tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J4: gst_phi_4prime5_j4,
                    tids.I_TENDON_TANGENCY_ANGLE_GST_4PRIME5_J5: gst_phi_4prime5_j5,
                    tids.I_TENDON_TANGENCY_ANGLE_GST_67_J6: gst_phi_67_j6,
                    tids.I_TENDON_TANGENCY_ANGLE_DFT_C5_J5: dft_phi_c5_j5,
                    tids.I_TENDON_TANGENCY_ANGLE_DFT_6C_J6: dft_phi_6c_j6,
                    tids.I_TENDON_TANGENCY_ANGLE_EDT1_5C_J5: edt1_phi_5c_jc,
                    tids.I_TENDON_TANGENCY_ANGLE_EDT2_56_J5: edt2_phi_56_j5,
                    tids.I_TENDON_TANGENCY_ANGLE_EDT2_67_J6: edt2_phi_6c_j6,
                },
                N_TENDON_TANGENCY_ANGLES,
            ),
            dim=1,
        )  # (B, N_TENDON_TANGENCY_ANGLES)

        # qs -> gst j3, j4, j5, j6; dft: j5, j6, edt1: none, edt2: j6-> 56-j6, 6c-j6,; kft: none
        # dft: q5 -> c5-j5, 56-j5, q6 -> 56-j6, 6c-j6;
        tendon_offsets_q_theta = torch.stack(
            list_from_dict(
                {
                    tids.I_Q_GST_3: gst_q_3_offset,
                    tids.I_Q_GST_4: gst_q_4_offset,
                    tids.I_Q_GST_5: gst_q_5_offset,
                    tids.I_Q_GST_6: gst_q_6_offset,
                    tids.I_Q_DFT_5: dft_q5_offset,
                    tids.I_Q_DFT_6: dft_q6_offset,
                },
                N_Q_OFFSETS,
            ),
            dim=1,
        )  # (B, N_Q_OFFSETS)
        tendon_offsets_qhat_thetahat = torch.stack(
            list_from_dict(
                {
                    tids.I_QHAT_EDT2_6: edt2_q6hat_offset,  # in relation to theta_6_hat
                },
                N_QHAT_OFFSETS,
            ),
            dim=1,
        )  # (B, N_QHAT_OFFSETS)

        tendon_offsets_theta = torch.stack(
            list_from_dict(
                {
                    tids.I_THETA_GST_3: joint_offsets_theta[:, tids.I_JOINT_3],
                    tids.I_THETA_GST_4: joint_offsets_theta[:, tids.I_JOINT_4],
                    tids.I_THETA_GST_5: joint_offsets_theta[:, tids.I_JOINT_5],
                    tids.I_THETA_ALL_6: joint_offsets_theta[:, tids.I_JOINT_6],
                    tids.I_THETA_DFT_5: dft_theta_offset_5,
                    tids.I_THETA_EDT1_4: edt1_theta_offset_4,
                    tids.I_THETA_EDT1_5: edt1_theta_offset_5,
                    tids.I_THETA_EDT2_4: edt2_theta_offset_4,
                    tids.I_THETA_EDT2_5: joint_offsets_theta[:, tids.I_JOINT_5],
                    tids.I_THETA_KFT_3: kft_theta_offset_3,
                    tids.I_THETA_KFT_8: joint_offsets_theta[:, tids.I_JOINT_8],
                },
                N_TENDON_THETA_OFFSETS,
            ),
            dim=1,
        )

        self.gst_stiffness = gst_stiffness
        self.gst_spring_rest_length = gst_spring_rest_length
        self.upper_gst_length = upper_gst_length
        self.lower_gst_length = lower_gst_length
        self.dft_stiffness = dft_stiffness
        self.dft_length = dft_length
        self.edt1_stiffness = edt1_stiffness
        self.edt1_length = edt1_length
        self.edt2_stiffness = edt2_stiffness
        self.edt2_length = edt2_length
        self.kft_stiffness = kft_stiffness
        self.kft_length = kft_length

        self.joint_directions = tc.joint_directions

        self.pulley_radii = pulley_radii
        self.pulley_radii_squared = pulley_radii ** 2
        self.link_lengths = torch.cat((chain_link_lengths, connector_link_lengths), dim=1)
        self.link_lengths_squared = self.link_lengths ** 2

        self.tendon_offsets_theta = tendon_offsets_theta
        self.tendon_offsets_q_theta = tendon_offsets_q_theta
        self.tendon_offsets_qhat_thetahat = tendon_offsets_qhat_thetahat
        self.tendon_section_lengths = tendon_section_lengths
        self.tendon_tangency_angles = tendon_tangency_angles

    def to_jit(self) -> TendonDataJIT:
        """Convert to JIT-compatible TendonDataJIT."""
        return TendonDataJIT(
            self.gst_stiffness,
            self.gst_spring_rest_length,
            self.upper_gst_length,
            self.lower_gst_length,
            self.dft_stiffness,
            self.dft_length,
            self.edt1_stiffness,
            self.edt1_length,
            self.edt2_stiffness,
            self.edt2_length,
            self.kft_stiffness,
            self.kft_length,
            self.joint_directions,
            self.pulley_radii,
            self.pulley_radii_squared,
            self.link_lengths,
            self.link_lengths_squared,
            self.tendon_offsets_theta,
            self.tendon_offsets_q_theta,
            self.tendon_offsets_qhat_thetahat,
            self.tendon_section_lengths,
            self.tendon_tangency_angles,
        )


def main():
    """Test tendon constants."""
    batch_size = 1
    tendon_data = TendonData(batch_size, dummy_randomization)
    print("Stiffness:", tendon_data.gst_stiffness)
    print("Spring rest length:", tendon_data.gst_spring_rest_length)
    print("Joint offsets (theta):", tendon_data.tendon_offsets_theta)
    print("Joint offsets (q-theta):", tendon_data.tendon_offsets_q_theta)
    print("Pulley radii:", tendon_data.pulley_radii)
    print("Link lengths:", tendon_data.link_lengths)
    print("Tendon section lengths:", tendon_data.tendon_section_lengths)
    print("Tendon tangency angles:", tendon_data.tendon_tangency_angles)
    print("Upper tendon length:", tendon_data.upper_gst_length)
    print("Lower tendon length:", tendon_data.lower_gst_length)
    print(
        "Phi_23:",
        torch.rad2deg(
            torch.atan2(
                tendon_data.pulley_radii[:, tids.I_RADIUS_GST_3],
                tendon_data.tendon_section_lengths[:, tids.I_TENDON_SECTION_LENGTH_GST_23],
            )
        ),
    )


if __name__ == "__main__":
    main()
