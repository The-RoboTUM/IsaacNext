"""Constants for our tendon model."""

from dataclasses import dataclass

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
    I_RADII_3: int = 0
    I_RADII_4: int = 1
    I_RADII_4prime: int = 2
    I_RADII_5: int = 3
    I_RADII_6: int = 4
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


@dataclass
class TendonConstants:
    """Fixed baseline mathematical constants for our tendon model: link lengths and pulley radii etc."""

    stiffness: float = 128e3
    spring_rest_length: float = 0.06
    upper_tendon_length: float = 0.5  # TODO: ask HW people
    lower_tendon_length: float = 0.5  # TODO: ask HW people
    joint_offsets_theta: list[float] = list_from_dict(
        {
            tids.I_JOINT_3: 227.671,
            tids.I_JOINT_4: 225.931,
            tids.I_JOINT_5: 200.0,
            tids.I_JOINT_6: 240.0,
        },
        N_JOINTS,
    )
    joint_directions: list[float] = list_from_dict(
        {
            tids.I_JOINT_3: -1.0,
            tids.I_JOINT_4: -1.0,
            tids.I_JOINT_5: -1.0,
            tids.I_JOINT_6: 1.0,
        },
        N_JOINTS,
    )
    pulley_radii: list[float] = list_from_dict(
        {
            tids.I_RADII_3: 0.0075,
            tids.I_RADII_4: 0.1,
            tids.I_RADII_4prime: 0.05,
            tids.I_RADII_5: 0.04,
            tids.I_RADII_6: 0.01,
        },
        N_RADII,
    )
    link_lengths: list[float] = list_from_dict(
        {
            tids.I_LINK_23: 0.33,
            tids.I_LINK_34: 0.461,
            tids.I_LINK_4prime5: 0.357,
            tids.I_LINK_56: 0.165,
            tids.I_LINK_67: 0.044,
        },
        N_LINKS,
    )
    length_2prime3: float = 0.0  # TODO: Meausre, distance from end of spring to joint 3


joint_offsets_q = (
    torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=dev),
)  # TODO: fill in => can be computed: theta = qleft + q + qright
tendon_section_lengths = (
    torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]], device=dev),
)  # TODO: fill in (compute from CAD)
tendon_tangency_angles = (
    torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=dev),
)  # TODO: fill in (compute from CAD)
# self.pulley_radii_squared = pulley_radii**2
# self.link_lengths_squared = link_lengths**2


class TendonData:
    """Tendon data for for parallel training.

    Includes randomization, derived constants, and batching.
    """

    def __init__(
        self, batch_size: int, randomization_ranges: dict[str, tuple[float, float]]
    ) -> None:
        pass
