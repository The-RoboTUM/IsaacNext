"""This script demonstrates applying random forces to a two-bar robot and visualizing them with markers using IsaacLab API."""

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/linus/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/cudnn/lib
print("Started")

import argparse
import json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Apply random forces to a two-bar robot and visualize them."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import torch
import numpy as np
import carb
import time
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_inv, quat_apply

# Enable anomaly detection for autograd
# torch.autograd.set_detect_anomaly(True)
# usd_path = "/media/C/Programmieren/RoboTUM/leg.usd"
usd_path = "/home/linus/IsaacNext/assets/Leg_free/leg.usd"


def get_leg_cfg() -> ArticulationCfg:
    """Returns an ArticulationCfg for the leg."""
    return ArticulationCfg(
        prim_path="/World/Bot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        actuators={
            # Spring: Rest 63,5mm, Compressed 20mm, => travel 43,5mm 128 N/mm
            "pantograph": ImplicitActuatorCfg(
                joint_names_expr=[
                    "rp1_pantograph",
                ],
                effort_limit_sim=1e9,
                velocity_limit_sim=1000.0,
                stiffness=128e3,
                damping=10.0,
            ),
            "hip_swing": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r2_pseudo_acetabulofemoral_flexion",
                ],
                effort_limit_sim=1.0e9,
                velocity_limit_sim=100.0,
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )


# Running pipeline:
# 0) transform joint angles to thetas and qs
# 1) evaluate conditions
# 2) compute energy with conditional function
# 3) differentiate w.r.t. joint angles
# 4) apply torques

# TODO: Measure tendon lengths, include angle at 4-4' pulley!
# FIXME: Fix acos/asin so that it always returns the right value (or have to use 2pi-phi)
# Change USD so that thetas are always between 0 and 2pi
class GSTTendonManager:
    # affects joints j3, j4, j5, j6 through links s23, s34, s45, s56, s67
    def __init__(
        self,
        robot: Articulation,
        stiffness: torch.Tensor,  # K spring
        spring_rest_length: torch.Tensor,  # L_R^S
        upper_tendon_length: torch.Tensor,  # L^UT
        lower_tendon_length: torch.Tensor,  # L^LT -> be sure to include the offset such that UT and LT
        #                                      are connected at the same angle at the pulley
        joint_offsets_q: torch.Tensor,  # q for joint_angle=0
        joint_offsets_theta: torch.Tensor,  # theta for joint_angle=0
        joint_directions: torch.Tensor,  # +-1 => joint_directions * joint_angles + joint_offsets_*
        pulley_radii: torch.Tensor,  # r3, r4, r4', r5, r6
        link_lengths: torch.Tensor,  # s23, s34, s45, s56, s67
        tendon_section_lengths: torch.Tensor,  # l23, l34, l4'5, l56, l67
        tendon_tangency_angles: torch.Tensor,  # phi_34^j4, phi_45^j4, phi_45^j5, phi_67^j6
        link_names: list[str],  # s23, s34, s45, s56, s67
        joint_names: list[str],  # j3, j4, j5, j6
    ):
        self.robot = robot
        self.device = robot.device
        self.link_names = link_names
        self.link_indices, _ = self.robot.find_bodies(
            self.link_names, preserve_order=True
        )
        self.joint_names = joint_names
        self.joint_indices, _ = self.robot.find_joints(
            self.joint_names, preserve_order=True
        )
        # TODO: add explanation in params to discuss these indices definitions
        self.LINK_LENGTHS_23 = 0
        self.LINK_LENGTHS_34 = 1
        self.LINK_LENGTHS_4prime5 = 2
        self.LINK_LENGTHS_56 = 3
        self.LINK_LENGTHS_67 = 4
        self.N_LINKS = len(self.link_indices)
        self.JOINT_ANGLES_3 = 0
        self.JOINT_ANGLES_4 = 1
        self.JOINT_ANGLES_5 = 2
        self.JOINT_ANGLES_6 = 3
        self.RADII_3 = 0
        self.RADII_4 = 1
        self.RADII_4prime = 2
        self.RADII_5 = 3
        self.RADII_6 = 4
        self.TENDON_TANGENGY_ANGLES_34_j4 = 0
        self.TENDON_TANGENGY_ANGLES_45_j4 = 1
        self.TENDON_TANGENGY_ANGLES_45_j5 = 2
        self.TENDON_TANGENGY_ANGLES_67_j6 = 3

        self.stiffness = stiffness
        self.spring_rest_length = spring_rest_length
        self.upper_tendon_length = upper_tendon_length
        self.lower_tendon_length = lower_tendon_length
        self.joint_offsets_q = joint_offsets_q
        self.joint_offsets_theta = joint_offsets_theta
        self.joint_directions = joint_directions
        self.pulley_radii = pulley_radii
        self.pulley_radii_squared = pulley_radii**2
        self.link_lengths = link_lengths
        self.link_lengths_squared = link_lengths**2
        self.tendon_section_lengths = tendon_section_lengths
        self.tendon_tangency_angles = tendon_tangency_angles

    def apply(self):
        joint_angles = (
            self.robot.data.joint_pos[:, self.joint_indices]
            .clone()
            .requires_grad_(True)
        )  # q3, q4, q5, q6
        # 0) transform joint angles to thetas and qs
        joint_angles_signed = self.joint_directions * joint_angles
        thetas = joint_angles_signed + self.joint_offsets_theta
        qs = joint_angles_signed + self.joint_offsets_q

        # 1) evaluate conditions
        # 1a) compute h5^B
        x_4prime6_squared = (
            self.link_lengths_squared[:, self.LINK_LENGTHS_4prime5]
            + self.link_lengths_squared[:, self.LINK_LENGTHS_56]
            - 2
            * self.link_lengths[:, self.LINK_LENGTHS_4prime5]
            * self.link_lengths[:, self.LINK_LENGTHS_56]
            * torch.cos(thetas[:, self.JOINT_ANGLES_5])
        )
        x_4prime6 = torch.sqrt(x_4prime6_squared)
        l_4prime6_squared = (
            x_4prime6_squared
            - (
                self.pulley_radii[:, self.RADII_4prime]
                - self.pulley_radii[:, self.RADII_6]
            )
            ** 2
        )
        l_4prime6 = torch.sqrt(l_4prime6_squared)
        phi_4prime_a_unsigned = torch.acos(
            (self.link_lengths_squared[:, self.LINK_LENGTHS_4prime5]
            + x_4prime6_squared
            - self.link_lengths_squared[:, self.LINK_LENGTHS_56])
            / (2 * self.link_lengths_squared[:, self.LINK_LENGTHS_4prime5] * x_4prime6)
        )
        phi_4prime_a = torch.where(
            thetas[:, self.JOINT_ANGLES_5] <= torch.pi,
            phi_4prime_a_unsigned,
            -phi_4prime_a_unsigned
        )
        phi_4prime_b = torch.acos(
            (
                self.pulley_radii_squared[:, self.RADII_4prime]
                + x_4prime6_squared
                - self.pulley_radii_squared[:, self.RADII_6]
                - l_4prime6_squared
            )
            / (2 * self.pulley_radii[:, self.RADII_4prime] * x_4prime6)
        )
        phi_4prime_B = phi_4prime_a + phi_4prime_b
        h5_B = (
            self.pulley_radii[:, self.RADII_4prime]
            - self.link_lengths[:, self.LINK_LENGTHS_4prime5] 
            * torch.cos(phi_4prime_B)
         )

        # 1b) compute h5^C and h6^C
        theta_6_a = torch.pi - thetas[:, self.JOINT_ANGLES_5] - phi_4prime_a
        theta_6_b = thetas[:, self.JOINT_ANGLES_6] - theta_6_a
        x_4prime7_squared = (
            x_4prime6_squared
            + self.link_lengths_squared[:, self.LINK_LENGTHS_67]
            - 2
            * x_4prime6
            * self.link_lengths[:, self.LINK_LENGTHS_67]
            * torch.cos(theta_6_b)
        )
        x_4prime7 = torch.sqrt(x_4prime7_squared)
        phi_4prime_d_unsigned = torch.acos(
            (x_4prime7_squared
            + x_4prime6_squared
            - self.link_lengths_squared[:, self.LINK_LENGTHS_67])
            / (2 * x_4prime6 * x_4prime7)
        )
        phi_4prime_d = torch.where(
            theta_6_b <= torch.pi,
            phi_4prime_d_unsigned,
            -phi_4prime_d_unsigned
        )
        phi_4prime_c = torch.acos(self.pulley_radii[:, self.RADII_4prime] / x_4prime7)
        phi_4prime_C = phi_4prime_a + phi_4prime_c + phi_4prime_d
        h5_C = (
            self.pulley_radii[:, self.RADII_4prime]
            - self.link_lengths[:, self.LINK_LENGTHS_4prime5] 
            * torch.cos(phi_4prime_C)
         ) # NOTE: changed this computation
        h6_C = self.pulley_radii[:, self.RADII_4prime] - x_4prime6 * torch.cos(
            phi_4prime_c + phi_4prime_d
        )

        # 1c) compute h6^D
        x_57_squared = (
            self.link_lengths_squared[:, self.LINK_LENGTHS_56]
            + self.link_lengths_squared[:, self.LINK_LENGTHS_67]
            - 2
            * self.link_lengths[:, self.LINK_LENGTHS_56]
            * self.link_lengths[:, self.LINK_LENGTHS_67]
            * torch.cos(thetas[:, self.JOINT_ANGLES_6])
        )
        x_57 = torch.sqrt(x_57_squared)
        l_57_squared = x_57_squared - self.pulley_radii[:, self.RADII_5] ** 2
        l_57 = torch.sqrt(l_57_squared)
        phi_5_a_unsigned = torch.acos(
            (self.link_lengths_squared[:, self.LINK_LENGTHS_56]
            + x_57_squared
            - self.link_lengths_squared[:, self.LINK_LENGTHS_67])
            / (2 * self.link_lengths_squared[:, self.LINK_LENGTHS_56] * x_57)
        )
        phi_5_a = torch.where(
            thetas[:, self.JOINT_ANGLES_6] <= torch.pi,
            phi_5_a_unsigned,
            -phi_5_a_unsigned
        )
        phi_5_b = torch.acos(self.pulley_radii_squared[:, self.RADII_5] / x_57)
        phi_5_D = phi_5_a + phi_5_b
        h6_D = self.pulley_radii[:, self.RADII_5] - self.link_lengths[
            :, self.LINK_LENGTHS_56
        ] * torch.cos(phi_5_D)

        h5_B_disengaged = torch.where(
            h5_B > self.pulley_radii[:, self.RADII_5],
            True,
            False,
        )
        h5_C_disengaged = torch.where(
            h5_C > self.pulley_radii[:, self.RADII_5],
            True,
            False,
        )
        h6_C_disengaged = torch.where(
            h6_C > self.pulley_radii[:, self.RADII_6],
            True,
            False,
        )
        h6_D_disengaged = torch.where(
            h6_D > self.pulley_radii[:, self.RADII_6],
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
            self.tendon_section_lengths[state_A, self.LINK_LENGTHS_4prime5]
            + qs[state_A, self.JOINT_ANGLES_5]
            * self.pulley_radii[state_A, self.RADII_5]
            + self.tendon_section_lengths[state_A, self.LINK_LENGTHS_56]
            + qs[state_A, self.JOINT_ANGLES_6]
            * self.pulley_radii[state_A, self.RADII_6]
            + self.tendon_section_lengths[state_A, self.LINK_LENGTHS_67]
        )

        # state B
        q6_B = (
            thetas[state_B, self.JOINT_ANGLES_6]
            - self.tendon_tangency_angles[state_B, self.TENDON_TANGENGY_ANGLES_67_j6]
            - 2 * torch.pi
            + phi_4prime_B[state_B]
            + thetas[state_B, self.JOINT_ANGLES_5]
        )

        lower_tendon_state_length_after_4prime[state_B] = (
            l_4prime6[state_B]
            + q6_B * self.pulley_radii[state_B, self.RADII_6]
            + self.tendon_section_lengths[state_B, self.LINK_LENGTHS_67]
        )

        # state C
        l_4prime7_squared = (
            x_4prime7_squared[state_C]
            - self.pulley_radii[state_C, self.RADII_4prime] ** 2
        )
        l_4prime7 = torch.sqrt(l_4prime7_squared)
        lower_tendon_state_length_after_4prime[state_C] = l_4prime7

        # state D
        q5_D = (
            thetas[state_D, self.JOINT_ANGLES_5]
            - self.tendon_tangency_angles[state_D, self.TENDON_TANGENGY_ANGLES_45_j5]
            - phi_5_D[state_D]
        )
        lower_tendon_state_length_after_4prime[state_D] = (
            self.tendon_section_lengths[state_D, self.LINK_LENGTHS_4prime5]
            + q5_D * self.pulley_radii[state_D, self.RADII_5]
            + l_57[state_D]
        )

        q4prime = (
            self.lower_tendon_length - lower_tendon_state_length_after_4prime
        ) / self.pulley_radii[:, self.RADII_4prime]

        q4 = (
            thetas[:, self.JOINT_ANGLES_4]
            - q4prime
            - self.tendon_tangency_angles[:, self.TENDON_TANGENGY_ANGLES_34_j4]
            - self.tendon_tangency_angles[:, self.TENDON_TANGENGY_ANGLES_45_j4]
        )

        delta_L_s = (
            self.upper_tendon_length
            - self.spring_rest_length
            - self.tendon_section_lengths[:, self.LINK_LENGTHS_23]
            - qs[:, self.JOINT_ANGLES_3] * self.pulley_radii[:, self.RADII_3]
            - self.tendon_section_lengths[:, self.LINK_LENGTHS_34]
            - q4 * self.pulley_radii[:, self.RADII_4]
        )

        not_slack = delta_L_s > 0.0

        energy = 0.5 * self.stiffness * delta_L_s[not_slack] ** 2
        # 3) differentiate w.r.t. joint angles
        tendon_torques = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=joint_angles,
            create_graph=False,
            allow_unused=True,
        )[0]
        # 4) apply torques: with axis [0 1 0], to each link
        tendon_torques_full = torch.zeros(
            (joint_angles.shape[0], self.N_LINKS, 3), device=self.device
        )
        tendon_torques_full[:, :4, 2] = tendon_torques
        tendon_torques_full[:, 1:, 2] -= tendon_torques

        self.robot.set_external_force_and_torque(
            forces=torch.zeros(
                (joint_angles.shape[0], self.N_LINKS, 3), device=self.device
            ),
            torques=tendon_torques_full,
            body_ids=self.link_indices,
        )


joints = [
    "rp1_pantograph",  # pantograph, actuated but always set to 0.0, stiffness? blockhöhe?     "s12p_pantograph_spring_assy_topv2_1" -> "s12p_pantograph_spring_assy_botv1_1"
    "r2_pseudo_acetabulofemoral_flexion",  # j2 -> position control, stiffness? damping?       "outside_hip_v2_assyv28_1" -> "knee_assyv9_1"
    "r3b_femorotibial_back",  # not actuated (fourbar), between j2 and j3                      "knee_assyv9_1" -> "s12p_pantograph_spring_assy_topv2_1"
    "r3f_femorotibial_front",  # j3 -> torque control, applied alongside other tendon torques  "knee_assyv9_1" -> "s12_front_assyv6_1"
    "r4f_intertarsal_front",  # only shows the pulley position q4' -> fix                      "s12_front_assyv6_1" -> "main_gst_pully_assyv4_1"
    "r4b_intertarsal_back",  # not actuated (fourbar), above j4                                "s12p_pantograph_spring_assy_botv1_1" -> "s23_assyv18_1_virtual"
    "r4p_intertarsal_pulley",  # j4, not actuated but affected by tendon                       "s12_front_assyv6_1" -> "s23_assyv18_1"
    "r5_metatarsophalangeal",  # j5, not actuated but affected by tendon                       "s23_assyv18_1" -> "s34_foot_connector_assyv20_1"
    "r6_interphalangeal",  # j6, not actuated but affected by tendon                           "s34_foot_connector_assyv20_1" -> "s45_digit_assyv2_1"
    "virtual_s23_assyv18_1_anchor",  # necessary for the urdf exporter but not actuated        "s23_assyv18_1" -> "s23_assyv18_1_virtual"
]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # IsaacLab simulation setup
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim_cfg.dt = 0.0032
    t_total = 3.0
    sim = SimulationContext(sim_cfg)
    #  sim.set_camera_view([5.0, 0.0, 1.5], [0.0, 0.0, 1.0])  # type: ignore

    # Add ground and light
    sim_utils.GroundPlaneCfg().func(
        "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
    )
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg()
    )

    robot = Articulation(cfg=get_leg_cfg())

    sim.reset()
    robot.write_joint_state_to_sim(
        position=robot.data.default_joint_pos,
        velocity=robot.data.default_joint_vel,
    )
    robot.write_data_to_sim()
    sim.step()  # step once to load the robot
    robot.update(sim.get_physics_dt())
    time.sleep(1)

    # Tendon manager setup
    gst_tendon_manager = GSTTendonManager(
        robot,
        stiffness=torch.tensor([128e3], device=device),
        spring_rest_length=torch.tensor([0.06], device=device),
        upper_tendon_length=torch.tensor([0.5], device=device),  # TODO: ask HW people
        lower_tendon_length=torch.tensor([0.5], device=device),  # TODO: ask HW people
        joint_offsets_q=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0]], device=device
        ),  # TODO (Linus): fill in => can be computed: theta = qleft + q + qright
        joint_offsets_theta=torch.deg2rad(
            torch.tensor(
                [
                    [
                        227.671,
                        225.931,
                        200.0,
                        240.0,
                    ]
                ],
                device=device,
            )
        ),
        joint_directions=torch.tensor([[-1.0, -1.0, -1.0, 1.0]], device=device),
        pulley_radii=torch.tensor(
            [[0.0075, 0.1, 0.05, 0.04, 0.01]], device=device
        ),  # NOTE: assumes tendon thickness 0, maybe correct with small offset
        link_lengths=torch.tensor(
            [[0.33, 0.461, 0.357, 0.165, 0.044]], device=device
        ),  # lj23 is not needed but included for easier indexing
        tendon_section_lengths=torch.tensor(
            [[0.1, 0.1, 0.1, 0.1, 0.1]], device=device
        ),  # TODO (Pilar): compute with appendix formulas in constants.py
        tendon_tangency_angles=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0]], device=device
        ),  # TODO (Pilar): compute with appendix formulas in constants.py
        link_names=[
            "knee_assyv9_1",  # 23
            "s12_front_assyv6_1",  # 34
            "s23_assyv18_1",  # 4'5
            "s34_foot_connector_assyv20_1",  # 56
            "s45_digit_assyv2_1",  # 67
        ],
        joint_names=[
            "r3f_femorotibial_front",  # j3
            "r4p_intertarsal_pulley",  # j4
            "r5_metatarsophalangeal",  # j5
            "r6_interphalangeal",  # j6
        ],
    )

    while True:
        joint_pos = []
        for _ in range(int(t_total / sim.get_physics_dt())):
            gst_tendon_manager.apply()
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            joint_pos.append(robot.data.joint_pos[0].clone())
        with open("outputs/joint_pos_gst.json", "w") as f:
            json.dump([pos.tolist() for pos in joint_pos], f)

        sim.reset()
        robot.write_joint_state_to_sim(
            position=robot.data.default_joint_pos,
            velocity=robot.data.default_joint_vel,
        )
        robot.write_data_to_sim()
        sim.step()  # step once to load the robot
        robot.update(sim.get_physics_dt())

        time.sleep(1)

    simulation_app.close()


if __name__ == "__main__":
    main()
