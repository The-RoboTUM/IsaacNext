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

from isaaclab.tendons.controllers.cpg import BirdBotCPGLeg, CPGParams
import torch
import numpy as np
import carb
import time
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext

from isaaclab.tendons.legacy.constants_old import (
    joint_names_left,
    joint_names_right,
)
from isaaclab.tendons.legacy.gst_manager import GSTTendonManager

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
                stiffness=10000.0,
                damping=10.0,
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
    mode = "free_fall"  # or "gait_cycle"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # IsaacLab simulation setup
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim_cfg.dt = 0.0032
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
    joint_indices_right, _ = robot.find_joints(joint_names_right, preserve_order=True)
    joint_indices_left, _ = robot.find_joints(joint_names_left, preserve_order=True)

    # Tendon manager setup
    gst_tendon_manager = GSTTendonManager(robot)
    cpg_leg_params_left = CPGParams(phi0=-np.pi / 2, f_hz=2.5)
    cpg_leg_left = BirdBotCPGLeg(cpg_leg_params_left)
    cpg_leg_params_right = CPGParams(phi0=np.pi / 2, f_hz=2.5)
    cpg_leg_right = BirdBotCPGLeg(cpg_leg_params_right)

    # while True:
    joint_pos_left = []
    states_left = []
    ls_left = []
    all_thetas_left = []
    all_tendon_torques_left = []
    joint_pos_right = []
    states_right = []
    ls_right = []
    all_thetas_right = []
    all_tendon_torques_right = []
    # TODO: Model ground as p-controller for more realistic interaction (e.g. proportional to penetration depth)
    try:
        for iteration in range(int(t_total / sim.get_physics_dt())):
            t = iteration * sim.get_physics_dt()
            a, b, c, d, le, thetas, tendon_torques_left, tendon_torques_right = (
                gst_tendon_manager.apply_actuated(
                    hip_position=torch.tensor(
                        [0.0, 0.0],
                        dtype=torch.float32,
                    ),
                    knee_torque=torch.tensor(
                        [
                            20.0,
                            20.0,
                        ],  # TODO: better torque computation
                        dtype=torch.float32,
                    ),
                    virtual_ground_height=0.38,
                    apply_tendons=False,
                )
            )
            states_left.append(
                "s"
                if le[0].item() > 0
                else (
                    "b"
                    if b[0].item() > 0
                    else ("c" if c[0].item() > 0 else ("d" if d[0].item() > 0 else "s"))
                )
            )
            ls_left.append(le[0].item())
            all_thetas_left.append(thetas[0].detach().cpu().numpy().tolist())
            all_tendon_torques_left.append(
                tendon_torques_left[0].cpu().numpy().tolist()
            )
            states_right.append(
                "s"
                if le[1].item() > 0
                else (
                    "b"
                    if b[1].item() > 0
                    else ("c" if c[1].item() > 0 else ("d" if d[1].item() > 0 else "s"))
                )
            )
            ls_right.append(le[1].item())
            all_thetas_right.append(thetas[1].detach().cpu().numpy().tolist())
            all_tendon_torques_right.append(
                tendon_torques_right[0].cpu().numpy().tolist()
            )

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            joint_pos_left.append(robot.data.joint_pos[0, joint_indices_left].clone())
            joint_pos_right.append(robot.data.joint_pos[0, joint_indices_right].clone())
    except Exception as e:
        carb.log_error(f"An error occurred during simulation: {e}")
        with open("outputs/joint_pos_gst_left.json", "w") as f:
            json.dump([pos.tolist() for pos in joint_pos_left], f)
        with open("outputs/states_gst_left.json", "w") as f:
            json.dump(states_left, f)
        with open("outputs/lengths_gst_left.json", "w") as f:
            json.dump(ls_left, f)
        with open("outputs/thetas_gst_left.json", "w") as f:
            json.dump(all_thetas_left, f)
        with open("outputs/torques_gst_left.json", "w") as f:
            json.dump(all_tendon_torques_left, f)
        with open("outputs/joint_pos_gst_right.json", "w") as f:
            json.dump([pos.tolist() for pos in joint_pos_right], f)
        with open("outputs/states_gst_right.json", "w") as f:
            json.dump(states_right, f)
        with open("outputs/lengths_gst_right.json", "w") as f:
            json.dump(ls_right, f)
        with open("outputs/thetas_gst_right.json", "w") as f:
            json.dump(all_thetas_right, f)
        with open("outputs/torques_gst_right.json", "w") as f:
            json.dump(all_tendon_torques_right, f)
        raise e

    with open("outputs/joint_pos_gst_left.json", "w") as f:
        json.dump([pos.tolist() for pos in joint_pos_left], f)
    with open("outputs/states_gst_left.json", "w") as f:
        json.dump(states_left, f)
    with open("outputs/lengths_gst_left.json", "w") as f:
        json.dump(ls_left, f)
    with open("outputs/thetas_gst_left.json", "w") as f:
        json.dump(all_thetas_left, f)
    with open("outputs/joint_pos_gst_right.json", "w") as f:
        json.dump([pos.tolist() for pos in joint_pos_right], f)
    with open("outputs/torques_gst_left.json", "w") as f:
        json.dump(all_tendon_torques_left, f)
    with open("outputs/states_gst_right.json", "w") as f:
        json.dump(states_right, f)
    with open("outputs/lengths_gst_right.json", "w") as f:
        json.dump(ls_right, f)
    with open("outputs/thetas_gst_right.json", "w") as f:
        json.dump(all_thetas_right, f)
    with open("outputs/torques_gst_right.json", "w") as f:
        json.dump(all_tendon_torques_right, f)

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
