"""This script demonstrates applying random forces to a two-bar robot and visualizing them with markers using IsaacLab API."""

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/linus/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/cudnn/lib
print("Started")

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Apply random forces to a two-bar robot and visualize them."
)
parser.add_argument(
    "--record_video", action="store_true", help="Record video of the simulation."
)
parser.add_argument(
    "--video_output",
    type=str,
    default="outputs/simulation.mp4",
    help="Output path for the video.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import carb
import time
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext


from isaaclab.tendons.legacy.gst_manager import GSTTendonManager

# usd_path = "/media/C/Programmieren/RoboTUM/leg.usd"
usd_path = "symlinks/Forrest_URDF_no_self_collision/Forrest_URDF.usd"
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # IsaacLab simulation setup
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim_cfg.dt = 0.0032
    t_total = 2.0
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(  # pyright: ignore[reportAttributeAccessIssue]
        [2.0, 2.0, 2.0], [0.0, 0.0, 0.5]
    )  # Set camera view for recording

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

    try:
        for iteration in range(int(t_total / sim.get_physics_dt())):
            t = iteration * sim.get_physics_dt()
            gst_tendon_manager.apply_jit()

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

    except KeyboardInterrupt as e:
        carb.log_error(f"An error occurred during simulation: {e}")
        raise e

    print("Completed simulation.")
    simulation_app.close()


if __name__ == "__main__":
    main()
