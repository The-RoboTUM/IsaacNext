"""Constants for our tendon model."""

from dataclasses import dataclass, field

import numpy as np
import torch

dev = "cuda"
N_LINKS: int = 5
N_RADII: int = 5
N_JOINTS: int = 4
N_TENDON_TANGENCY_ANGLES: int = 4


def list_from_dict(d: dict, n: int) -> list:
    """Convert a dict of lists to a list of lists."""
    assert (
        min(d.keys()) == 0 and max(d.keys()) == n - 1 and len(set(d.keys())) == n
    ), "Dict keys must be consecutive integers starting from 0."
    return [d[k] for k in sorted(d.keys())]


@dataclass
class TendonIndices:
    """Indices for our tendon model."""

    I_LINK_23: int = 0
    I_LINK_34: int = 1
    I_LINK_4prime5: int = 2
    I_LINK_56: int = 3
    I_LINK_67: int = 4
    I_JOINT_3: int = 0
    I_JOINT_4: int = 1
    I_JOINT_5: int = 2
    I_JOINT_6: int = 3
    I_RADIUS_3: int = 0
    I_RADIUS_4: int = 1
    I_RADIUS_4prime: int = 2
    I_RADIUS_5: int = 3
    I_RADIUS_6: int = 4
    I_TENDON_TANGENGY_ANGLES_34_j4: int = 0
    I_TENDON_TANGENGY_ANGLES_45_j4: int = 1
    I_TENDON_TANGENGY_ANGLES_45_j5: int = 2
    I_TENDON_TANGENGY_ANGLES_67_j6: int = 3


tids = TendonIndices()

link_names = list_from_dict(
    {
        tids.I_LINK_23: "knee_assyv9_1",  # 23
        tids.I_LINK_34: "s12_front_assyv6_1",  # 34
        tids.I_LINK_4prime5: "s23_assyv18_1",  # 4'5
        tids.I_LINK_56: "s34_foot_connector_assyv20_1",  # 56
        tids.I_LINK_67: "s45_digit_assyv2_1",  # 67
    },
    N_LINKS,
)
joint_names = list_from_dict(
    {
        tids.I_JOINT_3: "r3f_femorotibial_front",  # j3
        tids.I_JOINT_4: "r4p_intertarsal_pulley",  # j4
        tids.I_JOINT_5: "r5_metatarsophalangeal",  # j5
        tids.I_JOINT_6: "r6_interphalangeal",  # j6
    },
    N_JOINTS,
)


# TODO: verify tendon lengths when prototype is built
@dataclass
class TendonConstants:
    """Fixed baseline mathematical constants for our tendon model: link lengths and pulley radii etc."""

    stiffness: float = 128e3
    spring_rest_length: float = 0.06
    joint_offsets_theta: torch.Tensor = torch.deg2rad(
        torch.tensor(
            list_from_dict(
                {
                    tids.I_JOINT_3: 227.671,
                    tids.I_JOINT_4: 225.931,
                    tids.I_JOINT_5: 200.0,
                    tids.I_JOINT_6: 240.0,
                },
                N_JOINTS,
            ),
            device=dev,
        )
    )
    joint_directions: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_JOINT_3: -1.0,
                tids.I_JOINT_4: -1.0,
                tids.I_JOINT_5: -1.0,
                tids.I_JOINT_6: 1.0,
            },
            N_JOINTS,
        ),
        device=dev,
    )
    pulley_radii: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_RADIUS_3: 0.029,
                tids.I_RADIUS_4: 0.1,
                tids.I_RADIUS_4prime: 0.05,
                tids.I_RADIUS_5: 0.04,
                tids.I_RADIUS_6: 0.01,
            },
            N_RADII,
        ),
        device=dev,
    )
    link_lengths: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.I_LINK_23: 0.33,
                tids.I_LINK_34: 0.461,
                tids.I_LINK_4prime5: 0.357,
                tids.I_LINK_56: 0.165,
                tids.I_LINK_67: 0.044,
            },
            N_LINKS,
        ),
        device=dev,
    )
    b_23 = 0.11104  # distance from end of spring to joint 3
    a_23 = 0.0594097  # distance from other attachment point of pantograph on l23 to end of spring
    c_23 = 0.14  # distance from joint 3 to other attachment point of pantograph on l23
    angle_4prime5_to_j44prime = np.deg2rad(
        124.069
    )  # angle between link 4'5 and line from joint 4 to 4-4' transition


@dataclass
class TendonConstantRandomizationRanges:
    """Ranges for randomizing tendon constants for sim-to-real transfer."""

    stiffness: tuple[float, float] = (-10e3, 10e3)
    spring_rest_length: tuple[float, float] = (-0.005, 0.005)

    upper_tendon_length: tuple[float, float] = (-0.01, 0.01)
    lower_tendon_length: tuple[float, float] = (-0.01, 0.01)

    joint_offsets_theta: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_JOINT_3: (-0.05, 0.05),
                tids.I_JOINT_4: (-0.05, 0.05),
                tids.I_JOINT_5: (-0.05, 0.05),
                tids.I_JOINT_6: (-0.05, 0.05),
            },
            N_JOINTS,
        )
    )

    pulley_radii: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_RADIUS_3: (-0.001, 0.001),
                tids.I_RADIUS_4: (-0.001, 0.001),
                tids.I_RADIUS_4prime: (-0.001, 0.001),
                tids.I_RADIUS_5: (-0.001, 0.001),
                tids.I_RADIUS_6: (-0.001, 0.001),
            },
            N_RADII,
        )
    )

    link_lengths: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_LINK_23: (-0.001, 0.001),
                tids.I_LINK_34: (-0.001, 0.001),
                tids.I_LINK_4prime5: (-0.001, 0.001),
                tids.I_LINK_56: (-0.001, 0.001),
                tids.I_LINK_67: (-0.001, 0.001),
            },
            N_LINKS,
        )
    )
    b_23: tuple[float, float] = (-0.001, 0.001)
    a_23: tuple[float, float] = (-0.001, 0.001)
    c_23: tuple[float, float] = (-0.001, 0.001)
    angle_4prime5_to_j44prime: tuple[float, float] = (-0.05, 0.05)


dummy_randomization = TendonConstantRandomizationRanges(
    stiffness=(0.0, 0.0),
    spring_rest_length=(0.0, 0.0),
    upper_tendon_length=(0.0, 0.0),
    lower_tendon_length=(0.0, 0.0),
    joint_offsets_theta=list_from_dict(
        {
            tids.I_JOINT_3: (0.0, 0.0),
            tids.I_JOINT_4: (0.0, 0.0),
            tids.I_JOINT_5: (0.0, 0.0),
            tids.I_JOINT_6: (0.0, 0.0),
        },
        N_JOINTS,
    ),
    pulley_radii=list_from_dict(
        {
            tids.I_RADIUS_3: (0.0, 0.0),
            tids.I_RADIUS_4: (0.0, 0.0),
            tids.I_RADIUS_4prime: (0.0, 0.0),
            tids.I_RADIUS_5: (0.0, 0.0),
            tids.I_RADIUS_6: (0.0, 0.0),
        },
        N_RADII,
    ),
    link_lengths=list_from_dict(
        {
            tids.I_LINK_23: (0.0, 0.0),
            tids.I_LINK_34: (0.0, 0.0),
            tids.I_LINK_4prime5: (0.0, 0.0),
            tids.I_LINK_56: (0.0, 0.0),
            tids.I_LINK_67: (0.0, 0.0),
        },
        N_LINKS,
    ),
    b_23=(0.0, 0.0),
    a_23=(0.0, 0.0),
    c_23=(0.0, 0.0),
    angle_4prime5_to_j44prime=(0.0, 0.0),
)


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


class TendonData:
    """Tendon data for for parallel training.

    Includes randomization, derived constants, and batching.
    """

    def __init__(
        self, batch_size: int, randomization_ranges: TendonConstantRandomizationRanges
    ) -> None:
        """Initialize tendon data."""
        tc = TendonConstants()
        stiffness = tc.stiffness + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.stiffness
        )
        spring_rest_length = tc.spring_rest_length + torch.empty(
            batch_size, device=dev
        ).uniform_(*randomization_ranges.spring_rest_length)

        joint_offsets_theta = torch.stack(
            [
                tc.joint_offsets_theta[i]
                + torch.empty(batch_size, device=dev).uniform_(
                    *randomization_ranges.joint_offsets_theta[i]
                )
                for i in range(N_JOINTS)
            ],
            dim=1,
        )  # (B, N_JOINTS)

        pulley_radii = torch.stack(
            [
                torch.tensor(tc.pulley_radii[i], device=dev)
                + torch.empty(batch_size, device=dev).uniform_(
                    *randomization_ranges.pulley_radii[i]
                )
                for i in range(N_RADII)
            ],
            dim=1,
        )  # (B, N_RADII)

        link_lengths = torch.stack(
            [
                torch.tensor(tc.link_lengths[i], device=dev)
                + torch.empty(batch_size, device=dev).uniform_(
                    *randomization_ranges.link_lengths[i]
                )
                for i in range(N_LINKS)
            ],
            dim=1,
        )  # (B, N_LINKS)
        a_23 = tc.a_23 + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.a_23
        )
        b_23 = tc.b_23 + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.b_23
        )
        c_23 = tc.c_23 + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.c_23
        )
        angle_4prime5_to_j44prime = tc.angle_4prime5_to_j44prime + torch.empty(
            batch_size, device=dev
        ).uniform_(*randomization_ranges.angle_4prime5_to_j44prime)

        l_2prime3 = torch.sqrt(b_23**2 - pulley_radii[:, tids.I_RADIUS_3] ** 2)
        phi_23_j3_upper = torch.asin(
            pulley_radii[:, tids.I_RADIUS_3] / b_23
        )  # asin is okay because of right triangle
        phi_23_j3_lower = torch.acos((b_23**2 + c_23**2 - a_23**2) / (2 * b_23 * c_23))
        phi_23_j3 = phi_23_j3_upper + phi_23_j3_lower

        phi_34_j3, phi_34_j4, l_34 = opposite_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_3],
            pulley_radii[:, tids.I_RADIUS_4],
            link_lengths[:, tids.I_LINK_34],
        )

        phi_4prime5_j4, phi_4prime5_j5, l_4prime5 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_4prime],
            pulley_radii[:, tids.I_RADIUS_5],
            link_lengths[:, tids.I_LINK_4prime5],
        )

        phi_56_j5, phi_56_j6, l_56 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_5],
            pulley_radii[:, tids.I_RADIUS_6],
            link_lengths[:, tids.I_LINK_56],
        )

        phi_67_j6, _, l_67 = same_sided_wrap(
            pulley_radii[:, tids.I_RADIUS_6],
            torch.tensor(0.0, device=dev),
            link_lengths[:, tids.I_LINK_67],
        )

        tendon_section_lengths = torch.stack(
            list_from_dict(
                {
                    tids.I_LINK_23: l_2prime3,
                    tids.I_LINK_34: l_34,
                    tids.I_LINK_4prime5: l_4prime5,
                    tids.I_LINK_56: l_56,
                    tids.I_LINK_67: l_67,
                },
                N_LINKS,
            ),
            dim=1,
        )  # (B, N_LINKS)

        tendon_tangency_angles = torch.stack(
            list_from_dict(
                {
                    tids.I_TENDON_TANGENGY_ANGLES_34_j4: phi_34_j4,
                    tids.I_TENDON_TANGENGY_ANGLES_45_j4: phi_4prime5_j4,
                    tids.I_TENDON_TANGENGY_ANGLES_45_j5: phi_4prime5_j5,
                    tids.I_TENDON_TANGENGY_ANGLES_67_j6: phi_67_j6,
                },
                N_TENDON_TANGENCY_ANGLES,
            ),
            dim=1,
        )  # (B, N_TENDON_TANGENCY_ANGLES)

        q_3_offset = joint_offsets_theta[:, tids.I_JOINT_3] - phi_23_j3 - phi_34_j3
        q_4_offset = (
            joint_offsets_theta[:, tids.I_JOINT_4]
            - angle_4prime5_to_j44prime
            - phi_34_j4
        )

        q_4prime_relaxed = angle_4prime5_to_j44prime - phi_4prime5_j4
        q_5_offset = joint_offsets_theta[:, tids.I_JOINT_5] - phi_4prime5_j5 - phi_56_j5
        q_6_offset = joint_offsets_theta[:, tids.I_JOINT_6] - phi_56_j6 - phi_67_j6

        joint_offsets_q = torch.stack(
            list_from_dict(
                {
                    tids.I_JOINT_3: q_3_offset,
                    tids.I_JOINT_4: q_4_offset,
                    tids.I_JOINT_5: q_5_offset,
                    tids.I_JOINT_6: q_6_offset,
                },
                N_JOINTS,
            ),
            dim=1,
        )  # (B, N_JOINTS)

        upper_tendon_length = (
            spring_rest_length
            + l_2prime3
            + l_34
            + pulley_radii[:, tids.I_RADIUS_3] * q_3_offset
            + pulley_radii[:, tids.I_RADIUS_4] * q_4_offset
        )
        lower_tendon_length = (
            l_4prime5
            + l_56
            + l_67
            + pulley_radii[:, tids.I_RADIUS_4prime] * q_4prime_relaxed
            + pulley_radii[:, tids.I_RADIUS_5] * q_5_offset
            + pulley_radii[:, tids.I_RADIUS_6] * q_6_offset
        )
        # Note: we randomize upper and lower tendon lengths after computing other offsets because of manufacturing tolerances.
        upper_tendon_length += torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.upper_tendon_length
        )
        lower_tendon_length += torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.lower_tendon_length
        )

        self.stiffness = stiffness
        self.spring_rest_length = spring_rest_length
        self.joint_offsets_theta = joint_offsets_theta
        self.joint_offsets_q = joint_offsets_q
        self.pulley_radii = pulley_radii
        self.link_lengths = link_lengths
        self.tendon_section_lengths = tendon_section_lengths
        self.tendon_tangency_angles = tendon_tangency_angles

        self.upper_tendon_length = upper_tendon_length
        self.lower_tendon_length = lower_tendon_length

        self.joint_directions = tc.joint_directions

        self.pulley_radii_squared = pulley_radii**2
        self.link_lengths_squared = link_lengths**2


def main():
    """Test tendon constants."""
    batch_size = 1
    tendon_data = TendonData(batch_size, dummy_randomization)
    print("Stiffness:", tendon_data.stiffness)
    print("Spring rest length:", tendon_data.spring_rest_length)
    print("Joint offsets (theta):", tendon_data.joint_offsets_theta)
    print("Joint offsets (q):", tendon_data.joint_offsets_q)
    print("Pulley radii:", tendon_data.pulley_radii)
    print("Link lengths:", tendon_data.link_lengths)
    print("Tendon section lengths:", tendon_data.tendon_section_lengths)
    print("Tendon tangency angles:", tendon_data.tendon_tangency_angles)
    print("Upper tendon length:", tendon_data.upper_tendon_length)
    print("Lower tendon length:", tendon_data.lower_tendon_length)


if __name__ == "__main__":
    main()
