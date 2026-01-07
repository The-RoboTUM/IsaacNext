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

from isaaclab.tendons.constants import (
    tids,
    TendonData,
    dummy_randomization,
    link_names,
    joint_names,
    N_LINKS,
)

# usd_path = "/media/C/Programmieren/RoboTUM/leg.usd"
usd_path = "/home/linus/IsaacNext/assets/Leg_free/leg.usd"

# throw error on NaN in backprop
torch.autograd.set_detect_anomaly(True)


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


def angle_from_sides(a, b, c, eps=1e-6):
    # s = 0.5 * (a + b + c)
    # area = torch.sqrt(torch.clamp(s * (s - a) * (s - b) * (s - c), min=0.0))

    # sin_theta = 2 * area / (a * b)
    # cos_theta = (a * a + b * b - c * c) / (2 * a * b)

    # theta = torch.atan2(sin_theta, cos_theta)
    # return theta
    cos_theta = (a * a + b * b - c * c) / (2 * a * b)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta + eps)

    theta = torch.atan2(sin_theta, cos_theta)
    return theta


def angle_from_sws(a, b, theta):
    x = a - b * torch.cos(theta)
    y = b * torch.sin(torch.minimum(theta, 2 * torch.pi - theta))
    return torch.atan2(y, x)


# Running pipeline:
# 0) transform joint angles to thetas and qs
# 1) evaluate conditions
# 2) compute energy with conditional function
# 3) differentiate w.r.t. joint angles
# 4) apply torques


# TODO: Compute or measure geometry parameters
# TODO: Measure tendon lengths, include angle at 4-4' pulley!
# TODO: look at acos/asin if it always returns the right value (or have to use 2pi-phi)
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
        print(
            "Triangle with sides:",
            self.tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            x_4prime6,
            self.tendon_data.link_lengths[:, tids.I_LINK_56],
        )
        print("Angle at link 5:", thetas[:, tids.I_JOINT_5])
        print(
            "Numerator of acos argument:",
            (
                self.tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
                + x_4prime6_squared
                - self.tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            ),
        )
        print(
            "Denominator of acos argument:",
            2 * self.tendon_data.link_lengths[:, tids.I_LINK_4prime5] * x_4prime6,
        )
        print(
            "Acos argument - 1:",
            (
                self.tendon_data.link_lengths_squared[:, tids.I_LINK_4prime5]
                + x_4prime6_squared
                - self.tendon_data.link_lengths_squared[:, tids.I_LINK_56]
            )
            / (2 * self.tendon_data.link_lengths[:, tids.I_LINK_4prime5] * x_4prime6)
            - 1,
        )
        # phi_4prime_a_unsigned = angle_from_sides(
        #     self.link_lengths[:, self.LINK_LENGTHS_4prime5],
        #     x_4prime6,
        #     self.link_lengths[:, self.LINK_LENGTHS_56],
        # )
        phi_4prime_a_unsigned = angle_from_sws(
            self.tendon_data.link_lengths[:, tids.I_LINK_4prime5],
            self.tendon_data.link_lengths[:, tids.I_LINK_56],
            thetas[:, tids.I_JOINT_5],
        )
        # phi_4prime_a_unsigned = torch.acos(
        #     (
        #         self.link_lengths_squared[:, self.LINK_LENGTHS_4prime5]
        #         + x_4prime6_squared
        #         - self.link_lengths_squared[:, self.LINK_LENGTHS_56]
        #     )
        #     / (2 * self.link_lengths[:, self.LINK_LENGTHS_4prime5] * x_4prime6)
        # )
        phi_4prime_a = (
            torch.where(  # NOTE: finished this computation, check if now correct
                thetas[:, tids.I_JOINT_5] <= torch.pi,
                phi_4prime_a_unsigned,
                -phi_4prime_a_unsigned,
            )
        )
        phi_4prime_b = torch.acos(
            (  # NOTE: changed signs in this computation
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
        )  # NOTE: changed this computation

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
        phi_4prime_d_unsigned = torch.acos(
            (
                x_4prime7_squared
                + x_4prime6_squared
                - self.tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            )
            / (2 * x_4prime6 * x_4prime7)
        )
        phi_4prime_d = (
            torch.where(  # NOTE: finished this computation, check if now correct
                theta_6_b <= torch.pi, phi_4prime_d_unsigned, -phi_4prime_d_unsigned
            )
        )
        phi_4prime_c = torch.acos(
            self.tendon_data.pulley_radii[:, tids.I_RADIUS_4prime] / x_4prime7
        )
        phi_4prime_C = phi_4prime_a + phi_4prime_c + phi_4prime_d
        h5_C = self.tendon_data.pulley_radii[
            :, tids.I_RADIUS_4prime
        ] - self.tendon_data.link_lengths[:, tids.I_LINK_4prime5] * torch.cos(
            phi_4prime_C
        )  # NOTE: changed this computation
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
        # print("l_56_squared:", self.link_lengths_squared[:, self.LINK_LENGTHS_56])
        # print("l_67_squared:", self.link_lengths_squared[:, self.LINK_LENGTHS_67])
        # print("x_57_squared", x_57_squared)
        # print(
        #     "Argument of acos:",
        #     (
        #         self.link_lengths_squared[:, self.LINK_LENGTHS_56]
        #         + x_57_squared
        #         - self.link_lengths_squared[:, self.LINK_LENGTHS_67]
        #     )
        #     / (2 * self.link_lengths_squared[:, self.LINK_LENGTHS_56] * x_57),
        # )
        # print(
        #     "Triangle with sides:",
        #     self.link_lengths[:, self.LINK_LENGTHS_56],
        #     x_57,
        #     self.link_lengths[:, self.LINK_LENGTHS_67],
        # )
        phi_5_a_unsigned = torch.acos(
            (
                self.tendon_data.link_lengths_squared[:, tids.I_LINK_56]
                + x_57_squared
                - self.tendon_data.link_lengths_squared[:, tids.I_LINK_67]
            )
            / (2 * self.tendon_data.link_lengths[:, tids.I_LINK_56] * x_57)
        )
        phi_5_a = torch.where(  # NOTE: finished this computation, check if now correct
            thetas[:, tids.I_JOINT_6] <= torch.pi,
            phi_5_a_unsigned,
            -phi_5_a_unsigned,
        )
        print("x_57:", x_57)
        print("pulley radius 5:", self.tendon_data.pulley_radii[:, tids.I_RADIUS_5])
        print(
            "Argument of acos for phi_5_b:",
            self.tendon_data.pulley_radii[:, tids.I_RADIUS_5] / x_57,
        )
        phi_5_b = torch.acos(self.tendon_data.pulley_radii[:, tids.I_RADIUS_5] / x_57)
        phi_5_D = phi_5_a + phi_5_b
        h6_D = self.tendon_data.pulley_radii[
            :, tids.I_RADIUS_5
        ] - self.tendon_data.link_lengths[:, tids.I_LINK_56] * torch.cos(phi_5_D)

        # FIXME: changed the logic here to match the documentation
        # seeing the state_* definitions below, it seemed like the logic is inverted
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

        not_slack = delta_L_s > 0.0

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
        tendon_torques_full[:, :4, 2] = tendon_torques
        tendon_torques_full[:, 1:, 2] -= tendon_torques

        self.robot.set_external_force_and_torque(
            forces=torch.zeros((joint_angles.shape[0], N_LINKS, 3), device=self.device),
            torques=tendon_torques_full,
            body_ids=self.link_indices,
        )

        print("Applied GST tendon model.")
        return state_A, state_B, state_C, state_D, not_slack


joints = [
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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # IsaacLab simulation setup
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim_cfg.dt = 0.0032
    t_total = 1.0
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
    gst_tendon_manager = GSTTendonManager(robot)

    # while True:
    joint_pos = []
    states = []
    try:
        for _ in range(int(t_total / sim.get_physics_dt())):
            a, b, c, d, not_slack = gst_tendon_manager.apply()
            states.append(
                "s"
                if not_slack.sum() == 0
                else (
                    "a"
                    if a.sum() > 0
                    else ("b" if b.sum() > 0 else ("c" if c.sum() > 0 else "d"))
                )
            )
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            joint_pos.append(robot.data.joint_pos[0].clone())
    except Exception as e:
        carb.log_error(f"An error occurred during simulation: {e}")
        with open("outputs/joint_pos_gst.json", "w") as f:
            json.dump([pos.tolist() for pos in joint_pos], f)
        with open("outputs/states_gst.json", "w") as f:
            json.dump(states, f)
        raise e

    # sim.reset()
    # robot.write_joint_state_to_sim(
    #     position=robot.data.default_joint_pos,
    #     velocity=robot.data.default_joint_vel,
    # )
    # robot.write_data_to_sim()
    # sim.step()  # step once to load the robot
    # robot.update(sim.get_physics_dt())

    # time.sleep(1)

    simulation_app.close()


if __name__ == "__main__":
    main()
