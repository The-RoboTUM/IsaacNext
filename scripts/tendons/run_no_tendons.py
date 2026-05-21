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

from isaaclab.tendons.cpg import BirdBotCPGLeg, CPGParams
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

from isaaclab.tendons.constants_old import (
    tids,
    TendonData,
    dummy_randomization,
    link_names_left,
    link_names_right,
    joint_names_left,
    joint_names_right,
    N_LINKS_PER_LEG,
)

# usd_path = "/media/C/Programmieren/RoboTUM/leg.usd"
usd_path = "/media/C/Programmieren/RoboTUM/forrest_full_static.usd"
# usd_path = "/home/linus/IsaacNext/assets/Leg_free_v2/leg.usd"

# throw error on NaN in backprop
# torch.autograd.set_detect_anomaly(True)

# todo: measure torque output and other leg


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
                    "lp1_pantograph",
                ],
                effort_limit_sim=1e9,
                velocity_limit_sim=1000.0,
                stiffness=128e3,
                damping=10.0,
            ),
            "hip_swing": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r2_pseudo_acetabulofemoral_flexion",
                    "l2_pseudo_acetabulofemoral_flexion",
                ],
                effort_limit_sim=1.0e9,
                velocity_limit_sim=100.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "hip_roll": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r0_acetabulofemoral_roll",
                    "l0_acetabulofemoral_roll",
                ],
                effort_limit_sim=1.0e9,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=10.0,
            ),
            "hip_lateral": ImplicitActuatorCfg(
                joint_names_expr=[
                    "r1_acetabulofemoral_lateral",
                    "l1_acetabulofemoral_lateral",
                ],
                effort_limit_sim=1.0e9,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=10.0,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "r5_metatarsophalangeal": np.deg2rad(-19.9, dtype=np.float32),
                "l5_metatarsophalangeal": np.deg2rad(-19.9, dtype=np.float32),
                "r6_interphalangeal": np.deg2rad(25.0, dtype=np.float32),
                "l6_interphalangeal": np.deg2rad(25.0, dtype=np.float32),
            }
        ),
    )


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
    joint_indices_right, _ = robot.find_joints(joint_names_right, preserve_order=True)
    joint_indices_left, _ = robot.find_joints(joint_names_left, preserve_order=True)

    link_indices_right, _ = robot.find_bodies(link_names_right, preserve_order=True)
    link_indices_left, _ = robot.find_bodies(link_names_left, preserve_order=True)

    try:
        for link_id in [
            tids.I_CONNECTOR_LINK_GST_23,
            tids.I_LINK_34,
            tids.I_LINK_4prime5,
            tids.I_LINK_56,
            tids.I_LINK_67,
        ]:
            carb.log_warn(f"Link id {link_id}")
            for iteration in range(int(t_total / sim.get_physics_dt())):
                torques = torch.zeros((1, 2, 3), device=device, dtype=torch.float32)
                torque = 20
                torques[0, 0, 0] = torque
                torques[0, 1, 0] = torque
                robot.set_external_force_and_torque(
                    forces=torch.zeros_like(torques),
                    torques=torques,
                    body_ids=[
                        link_indices_right[link_id],
                        link_indices_left[link_id],
                    ],
                )

                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())
            robot.write_joint_state_to_sim(
                position=robot.data.default_joint_pos,
                velocity=robot.data.default_joint_vel,
            )
            for iteration in range(int(t_total / sim.get_physics_dt())):
                torques = torch.zeros((1, 2, 3), device=device, dtype=torch.float32)
                robot.set_external_force_and_torque(
                    forces=torch.zeros_like(torques),
                    torques=torques,
                    body_ids=[
                        link_indices_right[link_id],
                        link_indices_left[link_id],
                    ],
                )
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())

    except Exception as e:
        carb.log_error(f"An error occurred during simulation: {e}")
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

    print("Completed simulation.")
    simulation_app.close()


if __name__ == "__main__":
    main()
