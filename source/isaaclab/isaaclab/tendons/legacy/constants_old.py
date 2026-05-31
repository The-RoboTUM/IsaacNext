"""Constants for our tendon model."""

from dataclasses import dataclass, field

import numpy as np
import torch
import isaaclab.tendons.models.analytic.indices as tids

dev = "cuda"


@torch.jit.script
class TendonDataJIT:
    """JIT-compatible tendon data container with only tensor attributes."""

    def __init__(
        self,
        gst_stiffness: torch.Tensor,
        dft_stiffness: torch.Tensor,
        gst_spring_rest_length: torch.Tensor,
        dft_length: torch.Tensor,
        edt1_length: torch.Tensor,
        edt2_length: torch.Tensor,
        joint_offsets_theta: torch.Tensor,
        joint_offsets_gst_q: torch.Tensor,
        joint_offsets_dft_q: torch.Tensor,
        joint_offsets_edt1_q: torch.Tensor,
        joint_offsets_edt2_q: torch.Tensor,
        pulley_radii_gst: torch.Tensor,
        pulley_radii_dft: torch.Tensor,
        pulley_radii_edt1: torch.Tensor,
        pulley_radii_edt2: torch.Tensor,
        link_lengths: torch.Tensor,
        gst_tendon_section_lengths: torch.Tensor,
        dft_tendon_section_lengths: torch.Tensor,
        gst_tendon_tangency_angles: torch.Tensor,
        dft_tendon_tangency_angles: torch.Tensor,
        upper_gst_length: torch.Tensor,
        lower_gst_length: torch.Tensor,
        joint_directions: torch.Tensor,
        pulley_radii_squared: torch.Tensor,
        link_lengths_squared: torch.Tensor,
    ) -> None:
        self.gst_stiffness = gst_stiffness
        self.gst_spring_rest_length = gst_spring_rest_length
        self.joint_offsets_theta = joint_offsets_theta
        self.joint_offsets_gst_q = joint_offsets_gst_q
        self.pulley_radii = pulley_radii_gst
        self.link_lengths = link_lengths
        self.gst_tendon_section_lengths = gst_tendon_section_lengths
        self.dft_tendon_section_lengths = dft_tendon_section_lengths
        self.gst_tendon_tangency_angles = gst_tendon_tangency_angles
        self.dft_tendon_tangency_angles = dft_tendon_tangency_angles
        self.upper_gst_length = upper_gst_length
        self.lower_gst_length = lower_gst_length
        self.dft_length = dft_length
        self.joint_directions = joint_directions
        self.pulley_radii_squared = pulley_radii_squared
        self.link_lengths_squared = link_lengths_squared


N_LINKS_PER_LEG: int = 5
N_RADII: int = 5
N_JOINTS: int = 4
N_TENDON_TANGENCY_ANGLES: int = 4
N_GST_TENDON_TANGENCY_ANGLES: int = 4

JOINT_AXIS_IDX = 0  # axis index for joint torques around x-axis


def list_from_dict(d: dict, n: int) -> list:
    """Convert a dict of lists to a list of lists."""
    assert (
        min(d.keys()) == 0 and max(d.keys()) == n - 1 and len(set(d.keys())) == n
    ), "Dict keys must be consecutive integers starting from 0."
    return [d[k] for k in sorted(d.keys())]


link_names_right = list_from_dict(
    {
        tids.I_LINK_23: "outside_hip_v2_assy_axialv21_1",  # "knee_assyv9_1",  # 23
        tids.I_LINK_34: "s12_front_assyv14_1",  # "s12_front_assyv6_1",  # 34
        tids.I_LINK_4prime5: "s23_assyv21_1",  # "s23_assyv18_1",  # 4'5
        tids.I_LINK_56: "s34_foot_connector_assyv23_1",  # "s34_foot_connector_assyv20_1",  # 56
        tids.I_LINK_67: "s45_digit_assyv2_1",  # "s45_digit_assyv2_1",  # 67
    },
    N_LINKS_PER_LEG,
)
link_names_left = list_from_dict(
    {
        tids.I_LINK_23: "outside_hip_v2_assy_axialv21_2",  # 23
        tids.I_LINK_34: "s12_front_assyv14_2",  # 34
        tids.I_LINK_4prime5: "s23_assyv21_2",  # 4'5
        tids.I_LINK_56: "s34_foot_connector_assyv23_2",  # 56
        tids.I_LINK_67: "s45_digit_assyv2_2",  # 67
    },
    N_LINKS_PER_LEG,
)
joint_names_right = list_from_dict(
    {
        tids.GST_I_Q_OFFSET_3: "r3f_femorotibial_front",  # j3
        tids.GST_I_Q_OFFSET_4: "r4f_intertarsal_front",  # j4
        tids.GST_I_Q_OFFSET_5: "r5_metatarsophalangeal",  # j5
        tids.GST_I_Q_OFFSET_6: "r6_interphalangeal",  # j6
    },
    N_JOINTS,
)
joint_names_left = list_from_dict(
    {
        tids.GST_I_Q_OFFSET_3: "l3f_femorotibial_front",  # j3
        tids.GST_I_Q_OFFSET_4: "l4f_intertarsal_front",  # j4
        tids.GST_I_Q_OFFSET_5: "l5_metatarsophalangeal",  # j5
        tids.GST_I_Q_OFFSET_6: "l6_interphalangeal",  # j6
    },
    N_JOINTS,
)

all_joint_names_right = [
    "r0_acetabulofemoral_roll,"  # j0, position/torque control
    "r1_acetabulofemoral_lateral",  # j1, position/torque control
    "rp1_pantograph",  # pantograph, actuated but always set to 0.0, stiffness? blockhöhe?     "s12p_pantograph_spring_assy_topv2_1" -> "s12p_pantograph_spring_assy_botv1_1"
    "r2_pseudo_acetabulofemoral_flexion",  # j2 -> position control, stiffness? damping?       "outside_hip_v2_assyv28_1" -> "knee_assyv9_1"
    "r3b_femorotibial_back",  # excluded from articulation (fourbar), between j2 and j3        "knee_assyv9_1" -> "s12p_pantograph_spring_assy_topv2_1"
    "r3f_femorotibial_front",  # j3 -> torque control, applied alongside other tendon torques  "knee_assyv9_1" -> "s12_front_assyv6_1"
    "r4f_intertarsal_front",  # only shows the pulley position q4' -> fix                      "s12_front_assyv6_1" -> "main_gst_pully_assyv4_1"
    "r4b_intertarsal_back",  # not actuated (fourbar), above j4                                "s12p_pantograph_spring_assy_botv1_1" -> "s23_assyv18_1_virtual"
    "r4p_intertarsal_pulley",  # j4, not actuated but affected by tendon                       "s12_front_assyv6_1" -> "s23_assyv18_1"
    "r5_metatarsophalangeal",  # j5, not actuated but affected by tendon                       "s23_assyv18_1" -> "s34_foot_connector_assyv20_1"
    "r6_interphalangeal",  # j6, not actuated but affected by tendon                           "s34_foot_connector_assyv20_1" -> "s45_digit_assyv2_1"
    "virtual_s23_assyv18_1_anchor",  # necessary for the urdf exporter but not actuated        "s23_assyv18_1" -> "s23_assyv18_1_virtual"
]


# TODO: verify tendon lengths when prototype is built
@dataclass
class TendonConstants:
    """Fixed baseline mathematical constants for our tendon model: link lengths and pulley radii etc."""

    gst_stiffness: float = 128e3
    gst_spring_rest_length: float = 0.06
    dft_stiffness: float = 100e4  # FIXME: find out real value
    gst_spring_rest_length: float = 0.06
    joint_offsets_theta: torch.Tensor = torch.deg2rad(
        torch.tensor(
            list_from_dict(
                {
                    tids.GST_I_Q_OFFSET_3: 227.671,
                    tids.GST_I_Q_OFFSET_4: 225.931,
                    tids.GST_I_Q_OFFSET_5: 180.0,
                    tids.GST_I_Q_OFFSET_6: 270.0,
                },
                N_JOINTS,
            ),
            device=dev,
        )
    )
    joint_directions: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.GST_I_Q_OFFSET_3: -1.0,
                tids.GST_I_Q_OFFSET_4: +1.0,
                tids.GST_I_Q_OFFSET_5: -1.0,
                tids.GST_I_Q_OFFSET_6: -1.0,
            },
            N_JOINTS,
        ),
        device=dev,
    )
    pulley_radii: torch.Tensor = torch.tensor(
        list_from_dict(
            {
                tids.GST_I_RADIUS_3: 0.029,
                tids.GST_I_RADIUS_4: 0.1,
                tids.GST_I_RADIUS_4prime: 0.05,
                tids.GST_I_RADIUS_5: 0.04,
                tids.GST_I_RADIUS_6: 0.01,
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
            N_LINKS_PER_LEG,
        ),
        device=dev,
    )
    b_23 = 0.11104  # distance from end of spring to joint 3
    a_23 = 0.0594097  # distance from other attachment point of pantograph on l23 to end of spring
    c_23 = 0.14  # distance from joint 3 to other attachment point of pantograph on l23
    angle_4prime5_to_j44prime = np.deg2rad(124.069)  # angle between link 4'5 and line from joint 4 to 4-4' transition
    dft_attachment_point_to_j5 = 0.08  # FIXME: measure correct value
    dft_limit_angle_theta5 = float(np.deg2rad(190))
    dft_limit_angle_theta6 = float(np.deg2rad(240))
    # distance from dft tendon attachment point to joint 5, along the line from joint 4 to joint 5


@dataclass
class TendonConstantRandomizationRanges:
    """Ranges for randomizing tendon constants for sim-to-real transfer."""

    gst_stiffness: tuple[float, float] = (-10e3, 10e3)
    dft_stiffness: tuple[float, float] = (-10e3, 10e3)
    gst_spring_rest_length: tuple[float, float] = (-0.005, 0.005)
    dft_length: tuple[float, float] = (-0.005, 0.005)

    upper_tendon_length: tuple[float, float] = (-0.01, 0.01)
    lower_tendon_length: tuple[float, float] = (-0.01, 0.01)

    joint_offsets_theta: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.GST_I_Q_OFFSET_3: (-0.05, 0.05),
                tids.GST_I_Q_OFFSET_4: (-0.05, 0.05),
                tids.GST_I_Q_OFFSET_5: (-0.05, 0.05),
                tids.GST_I_Q_OFFSET_6: (-0.05, 0.05),
            },
            N_JOINTS,
        )
    )

    pulley_radii: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.GST_I_RADIUS_3: (-0.001, 0.001),
                tids.GST_I_RADIUS_4: (-0.001, 0.001),
                tids.GST_I_RADIUS_4prime: (-0.001, 0.001),
                tids.GST_I_RADIUS_5: (-0.001, 0.001),
                tids.GST_I_RADIUS_6: (-0.001, 0.001),
            },
            N_RADII,
        )
    )

    link_lengths: list[tuple[float, float]] = field(
        default_factory=lambda: list_from_dict(
            {
                tids.I_CONNECTOR_LINK_GST_23: (-0.001, 0.001),
                tids.I_LINK_34: (-0.001, 0.001),
                tids.I_LINK_4prime5: (-0.001, 0.001),
                tids.I_LINK_56: (-0.001, 0.001),
                tids.GST_I_LINK_67: (-0.001, 0.001),
            },
            N_LINKS_PER_LEG,
        )
    )
    b_23: tuple[float, float] = (-0.001, 0.001)
    a_23: tuple[float, float] = (-0.001, 0.001)
    c_23: tuple[float, float] = (-0.001, 0.001)
    angle_4prime5_to_j44prime: tuple[float, float] = (-0.05, 0.05)
    dft_limit_angle_theta5: tuple[float, float] = (np.deg2rad(-5), np.deg2rad(5))
    dft_limit_angle_theta6: tuple[float, float] = (np.deg2rad(-5), np.deg2rad(5))


dummy_randomization = TendonConstantRandomizationRanges(
    gst_stiffness=(0.0, 0.0),
    dft_stiffness=(0.0, 0.0),
    gst_spring_rest_length=(0.0, 0.0),
    dft_length=(0.0, 0.0),
    upper_tendon_length=(0.0, 0.0),
    lower_tendon_length=(0.0, 0.0),
    joint_offsets_theta=list_from_dict(
        {
            tids.GST_I_Q_OFFSET_3: (0.0, 0.0),
            tids.GST_I_Q_OFFSET_4: (0.0, 0.0),
            tids.GST_I_Q_OFFSET_5: (0.0, 0.0),
            tids.GST_I_Q_OFFSET_6: (0.0, 0.0),
        },
        N_JOINTS,
    ),
    pulley_radii=list_from_dict(
        {
            tids.GST_I_RADIUS_3: (0.0, 0.0),
            tids.GST_I_RADIUS_4: (0.0, 0.0),
            tids.GST_I_RADIUS_4prime: (0.0, 0.0),
            tids.GST_I_RADIUS_5: (0.0, 0.0),
            tids.GST_I_RADIUS_6: (0.0, 0.0),
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
        N_LINKS_PER_LEG,
    ),
    b_23=(0.0, 0.0),
    a_23=(0.0, 0.0),
    c_23=(0.0, 0.0),
    angle_4prime5_to_j44prime=(0.0, 0.0),
    dft_limit_angle_theta5=(0.0, 0.0),
    dft_limit_angle_theta6=(0.0, 0.0),
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
        self,
        batch_size: int,
        randomization_ranges: TendonConstantRandomizationRanges,
    ) -> None:
        """Initialize tendon data."""
        batch_size *= 2  # for left and right legs
        tc = TendonConstants()

        joint_offsets_theta = torch.stack(
            [
                tc.joint_offsets_theta[i]
                + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.joint_offsets_theta[i])
                for i in range(N_JOINTS)
            ],
            dim=1,
        )  # (B, N_JOINTS)

        pulley_radii = torch.stack(
            [
                torch.tensor(tc.pulley_radii[i], device=dev)
                + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.pulley_radii[i])
                for i in range(N_RADII)
            ],
            dim=1,
        )  # (B, N_RADII)

        link_lengths = torch.stack(
            [
                torch.tensor(tc.link_lengths[i], device=dev)
                + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.link_lengths[i])
                for i in range(N_LINKS_PER_LEG)
            ],
            dim=1,
        )  # (B, N_LINKS)

        # ----------------------GST ------------------ #
        gst_stiffness = tc.gst_stiffness + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.gst_stiffness
        )
        gst_spring_rest_length = tc.gst_spring_rest_length + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.gst_spring_rest_length
        )
        a_23 = tc.a_23 + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.a_23)
        b_23 = tc.b_23 + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.b_23)
        c_23 = tc.c_23 + torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.c_23)
        gst_angle_4prime5_to_j44prime = tc.angle_4prime5_to_j44prime + torch.empty(batch_size, device=dev).uniform_(
            *randomization_ranges.angle_4prime5_to_j44prime
        )

        l_2prime3 = torch.sqrt(b_23**2 - pulley_radii[:, tids.GST_I_RADIUS_3] ** 2)
        phi_23_j3_upper = torch.acos(pulley_radii[:, tids.GST_I_RADIUS_3] / b_23)
        phi_23_j3_lower = torch.acos((b_23**2 + c_23**2 - a_23**2) / (2 * b_23 * c_23))
        phi_23_j3 = phi_23_j3_upper + phi_23_j3_lower

        phi_34_j3, phi_34_j4, l_34 = opposite_sided_wrap(
            pulley_radii[:, tids.GST_I_RADIUS_3],
            pulley_radii[:, tids.GST_I_RADIUS_4],
            link_lengths[:, tids.I_LINK_34],
        )

        phi_4prime5_j4, phi_4prime5_j5, l_4prime5 = same_sided_wrap(
            pulley_radii[:, tids.GST_I_RADIUS_4prime],
            pulley_radii[:, tids.GST_I_RADIUS_5],
            link_lengths[:, tids.I_LINK_4prime5],
        )

        phi_56_j5, phi_56_j6, l_56 = same_sided_wrap(
            pulley_radii[:, tids.GST_I_RADIUS_5],
            pulley_radii[:, tids.GST_I_RADIUS_6],
            link_lengths[:, tids.I_LINK_56],
        )

        phi_67_j6, _, l_67 = same_sided_wrap(
            pulley_radii[:, tids.GST_I_RADIUS_6],
            torch.tensor(0.0, device=dev),
            link_lengths[:, tids.GST_I_LINK_67],
        )

        gst_tendon_section_lengths = torch.stack(
            list_from_dict(
                {
                    tids.I_LINK_23: l_2prime3,
                    tids.I_LINK_34: l_34,
                    tids.I_LINK_4prime5: l_4prime5,
                    tids.I_LINK_56: l_56,
                    tids.I_LINK_67: l_67,
                },
                N_LINKS_PER_LEG,
            ),
            dim=1,
        )  # (B, N_LINKS)

        gst_tendon_tangency_angles = torch.stack(
            list_from_dict(
                {
                    tids.GST_I_TENDON_TANGENGY_ANGLES_34_J4: phi_34_j4,
                    tids.GST_I_TENDON_TANGENGY_ANGLES_4PRIME5_J4: phi_4prime5_j4,
                    tids.GST_I_TENDON_TANGENGY_ANGLES_4PRIME5_J5: phi_4prime5_j5,
                    tids.GST_I_TENDON_TANGENGY_ANGLES_67_J6: phi_67_j6,
                },
                N_GST_TENDON_TANGENCY_ANGLES,
            ),
            dim=1,
        )  # (B, N_TENDON_TANGENCY_ANGLES)
        print("phi_23_j3:", np.rad2deg(phi_23_j3[0].item()))
        gst_q_3_offset = joint_offsets_theta[:, tids.GST_I_Q_OFFSET_3] - phi_23_j3 - phi_34_j3
        gst_q_4_offset = joint_offsets_theta[:, tids.GST_I_Q_OFFSET_4] - gst_angle_4prime5_to_j44prime - phi_34_j4

        gst_q_4prime_relaxed = gst_angle_4prime5_to_j44prime - phi_4prime5_j4
        gst_q_5_offset = joint_offsets_theta[:, tids.GST_I_Q_OFFSET_5] - phi_4prime5_j5 - phi_56_j5
        gst_q_6_offset = joint_offsets_theta[:, tids.GST_I_Q_OFFSET_6] - phi_56_j6 - phi_67_j6

        joint_offsets_gst_q = torch.stack(
            list_from_dict(
                {
                    tids.GST_I_Q_OFFSET_3: gst_q_3_offset,
                    tids.GST_I_Q_OFFSET_4: gst_q_4_offset,
                    tids.GST_I_Q_OFFSET_5: gst_q_5_offset,
                    tids.GST_I_Q_OFFSET_6: gst_q_6_offset,
                },
                N_JOINTS,
            ),
            dim=1,
        )  # (B, N_JOINTS)

        upper_gst_length = (
            gst_spring_rest_length
            + l_2prime3
            + l_34
            + pulley_radii[:, tids.GST_I_RADIUS_3] * gst_q_3_offset
            + pulley_radii[:, tids.GST_I_RADIUS_4] * gst_q_4_offset
        )
        lower_gst_length = (
            l_4prime5
            + l_56
            + l_67
            + pulley_radii[:, tids.GST_I_RADIUS_4prime] * gst_q_4prime_relaxed
            + pulley_radii[:, tids.GST_I_RADIUS_5] * gst_q_5_offset
            + pulley_radii[:, tids.GST_I_RADIUS_6] * gst_q_6_offset
        )
        # Note: we randomize upper and lower tendon lengths after computing other offsets because of manufacturing tolerances.
        upper_gst_length += torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.upper_tendon_length)
        lower_gst_length += torch.empty(batch_size, device=dev).uniform_(*randomization_ranges.lower_tendon_length)

        # -------------------- DFT ------------------ #
        "dft_stiffness", "dft_length", "dft_tendon_section_lengths", "dft_tendon_tangency_angles"

        self.gst_stiffness = gst_stiffness
        self.gst_spring_rest_length = gst_spring_rest_length
        self.joint_offsets_theta = joint_offsets_theta
        self.joint_offsets_gst_q = joint_offsets_gst_q
        self.pulley_radii = pulley_radii
        self.link_lengths = link_lengths
        self.gst_tendon_section_lengths = gst_tendon_section_lengths
        self.gst_tendon_tangency_angles = gst_tendon_tangency_angles

        self.upper_gst_length = upper_gst_length
        self.lower_gst_length = lower_gst_length

        self.joint_directions = tc.joint_directions

        self.pulley_radii_squared = pulley_radii**2
        self.link_lengths_squared = link_lengths**2

    def to_jit(self) -> TendonDataJIT:
        """Convert to JIT-compatible TendonDataJIT."""
        return TendonDataJIT(
            gst_stiffness=self.gst_stiffness,
            dft_length=self.link_lengths,
            dft_stiffness=self.gst_stiffness,
            dft_tendon_section_lengths=self.gst_tendon_section_lengths,
            dft_tendon_tangency_angles=self.gst_tendon_tangency_angles,
            edt1_length=self.link_lengths,
            edt2_length=self.link_lengths,
            joint_offsets_dft_q=self.joint_offsets_gst_q,
            joint_offsets_edt1_q=self.joint_offsets_gst_q,
            joint_offsets_edt2_q=self.joint_offsets_gst_q,
            pulley_radii_dft=self.pulley_radii,
            pulley_radii_edt1=self.pulley_radii,
            pulley_radii_edt2=self.pulley_radii,
            gst_spring_rest_length=self.gst_spring_rest_length,
            joint_offsets_theta=self.joint_offsets_theta,
            joint_offsets_gst_q=self.joint_offsets_gst_q,
            pulley_radii_gst=self.pulley_radii,
            link_lengths=self.link_lengths,
            gst_tendon_section_lengths=self.gst_tendon_section_lengths,
            gst_tendon_tangency_angles=self.gst_tendon_tangency_angles,
            upper_gst_length=self.upper_gst_length,
            lower_gst_length=self.lower_gst_length,
            joint_directions=self.joint_directions,
            pulley_radii_squared=self.pulley_radii_squared,
            link_lengths_squared=self.link_lengths_squared,
        )


def main():
    """Test tendon constants."""
    batch_size = 1
    tendon_data = TendonData(batch_size, dummy_randomization)
    print("Stiffness:", tendon_data.gst_stiffness)
    print("Spring rest length:", tendon_data.gst_spring_rest_length)
    print("Joint offsets (theta):", tendon_data.joint_offsets_theta)
    print("Joint offsets (q):", tendon_data.joint_offsets_gst_q)
    print("Pulley radii:", tendon_data.pulley_radii)
    print("Link lengths:", tendon_data.link_lengths)
    print("Tendon section lengths:", tendon_data.gst_tendon_section_lengths)
    print("Tendon tangency angles:", tendon_data.gst_tendon_tangency_angles)
    print("Upper tendon length:", tendon_data.upper_gst_length)
    print("Lower tendon length:", tendon_data.lower_gst_length)
    print(
        "Phi_23:",
        torch.rad2deg(
            torch.atan2(
                tendon_data.pulley_radii[:, tids.GST_I_RADIUS_3],
                tendon_data.gst_tendon_section_lengths[:, tids.I_CONNECTOR_LINK_GST_23],
            )
        ),
    )
    print("l_23:", tendon_data.gst_tendon_section_lengths[:, tids.I_CONNECTOR_LINK_GST_23])


if __name__ == "__main__":
    main()
