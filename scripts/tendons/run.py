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
from isaaclab.tendons.gst_manager import GSTTendonManager

# usd_path = "/media/C/Programmieren/RoboTUM/leg.usd"
usd_path = "/home/linus/IsaacNext/assets/Leg_free_v2/leg.usd"

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


def main():
    mode = "free_fall"  # or "gait_cycle"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # IsaacLab simulation setup
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim_cfg.dt = 0.00032
    t_total = 2.0
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
    joint_indices, _ = robot.find_joints(joint_names, preserve_order=True)

    # Tendon manager setup
    gst_tendon_manager = GSTTendonManager(robot)

    # while True:
    joint_pos = []
    states = []
    ls = []
    all_thetas = []
    try:
        for iteration in range(int(t_total / sim.get_physics_dt())):
            a, b, c, d, not_slack, le, thetas = gst_tendon_manager.apply()
            states.append(
                "s"
                if not_slack.sum() == 0
                else (
                    "a"
                    if a.sum() > 0
                    else ("b" if b.sum() > 0 else ("c" if c.sum() > 0 else "d"))
                )
            )
            ls.append(le)
            all_thetas.append(thetas)
            if mode == "gait_cycle":
                robot.set_joint_position_target(
                    0.5
                    * np.pi
                    * torch.sin(
                        torch.tensor(
                            [2 * np.pi * 0.5 * iteration * sim.get_physics_dt()]
                        )
                    )
                )
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            joint_pos.append(robot.data.joint_pos[0, joint_indices].clone())
    except Exception as e:
        carb.log_error(f"An error occurred during simulation: {e}")
        with open("outputs/joint_pos_gst.json", "w") as f:
            json.dump([pos.tolist() for pos in joint_pos], f)
        with open("outputs/states_gst.json", "w") as f:
            json.dump(states, f)
        with open("outputs/lengths_gst.json", "w") as f:
            json.dump(ls, f)
        with open("outputs/thetas_gst.json", "w") as f:
            json.dump(all_thetas, f)
        raise e

    with open("outputs/joint_pos_gst.json", "w") as f:
        json.dump([pos.tolist() for pos in joint_pos], f)
    with open("outputs/states_gst.json", "w") as f:
        json.dump(states, f)
    with open("outputs/lengths_gst.json", "w") as f:
        json.dump(ls, f)
    with open("outputs/thetas_gst.json", "w") as f:
        json.dump(all_thetas, f)

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
