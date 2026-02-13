"""This script demonstrates applying random forces to a two-bar robot and visualizing them with markers using IsaacLab API."""

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/linus/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/cudnn/lib
print("Started")

import argparse
import json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Apply random forces to a two-bar robot and visualize them."
)
parser.add_argument(
    "--record_video", action="store_true", help="Record video of the simulation."
)
parser.add_argument("--jit", action="store_true", help="Whether to use jit.")
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

from isaaclab.tendons.cpg import BirdBotCPGLeg, CPGParams
import torch
import numpy as np
import cv2
import os
import carb
import time
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_inv, quat_apply
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg

from isaaclab.tendons.constants import (
    tids,
    TendonData,
    dummy_randomization,
    link_names_left,
    link_names_right,
    joint_names_left,
    joint_names_right,
    N_LINKS_PER_LEG,
)
from isaaclab.tendons.gst_manager import GSTTendonManager

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

    os.makedirs("outputs", exist_ok=True)
    if not args_cli.jit:
        with open(os.path.join("outputs", "gst_data_left.jsonl"), "w") as f:
            f.write("")
        with open(os.path.join("outputs", "gst_data_right.jsonl"), "w") as f:
            f.write("")

    def append_data(data, side):
        fd = os.open(
            os.path.join("outputs", f"gst_data_{side}.jsonl"),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        os.write(fd, (json.dumps(data) + "\n").encode())
        os.close(fd)

    # Setup camera for video recording
    video_writer = None
    camera = None
    if args_cli.record_video:
        # Create a camera prim at a fixed position viewing the robot
        import isaacsim.core.utils.prims as prim_utils

        prim_utils.create_prim("/World/Camera", "Xform")

        camera_cfg = TiledCameraCfg(
            prim_path="/World/Camera/RecordCamera",
            update_period=0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(2.0, 0.0, 1.0),
                rot=(0.0, 0.0, 0.0, 1.0),  # (-0.383, 0.0, 0.0, 0.924),
                convention="world",
            ),
        )
        camera = TiledCamera(camera_cfg)

        # Setup video writer
        os.makedirs(os.path.dirname(args_cli.video_output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(  # pyright: ignore[reportAttributeAccessIssue]
            *"mp4v"
        )
        fps = 30  # int(1.0 / sim_cfg.dt)
        video_writer = cv2.VideoWriter(args_cli.video_output, fourcc, fps, (1280, 720))

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
    phi_0_combined_offset = np.pi / 2  # np.pi / 2
    cpg_leg_params_left = CPGParams(phi0=-np.pi / 2 + phi_0_combined_offset, f_hz=2.5)
    cpg_leg_left = BirdBotCPGLeg(cpg_leg_params_left)
    cpg_leg_params_right = CPGParams(phi0=np.pi / 2 + phi_0_combined_offset, f_hz=2.5)
    cpg_leg_right = BirdBotCPGLeg(cpg_leg_params_right)

    try:
        for iteration in range(int(t_total / sim.get_physics_dt())):
            t = iteration * sim.get_physics_dt()
            kwargs = dict(
                hip_position=torch.tensor(
                    [
                        cpg_leg_left.hip_flex(t)[0],
                        cpg_leg_right.hip_flex(t)[0],
                    ],
                    dtype=torch.float32,
                ),
                knee_torque=torch.tensor(
                    [
                        cpg_leg_left.knee(t)[0] * 20.0,
                        cpg_leg_right.knee(t)[0] * 20.0,
                    ],  # TODO: better torque computation
                    dtype=torch.float32,
                ),
                virtual_ground_height=0.38,
                apply_tendons=True,
            )
            if not args_cli.jit:
                info = gst_tendon_manager.apply_actuated_debug(**kwargs)
                data_left = {}
                data_left["state"] = ("s" if info["delta_l"][0].item() > 0 else "") + (
                    "a"
                    if info["a"][0].item() > 0
                    else (
                        "b"
                        if info["b"][0].item() > 0
                        else (
                            "c"
                            if info["c"][0].item() > 0
                            else ("d" if info["d"][0].item() > 0 else "x")
                        )
                    )
                )
                data_left["delta_l"] = info["delta_l"][0].item()
                data_left["thetas"] = info["thetas"][0].detach().cpu().numpy().tolist()
                data_left["qs"] = info["qs"][0].detach().cpu().numpy().tolist()
                data_left["q4"] = info["q4"][0].item()
                data_left["q4prime"] = info["q4prime"][0].item()
                data_left["q5_D"] = info["q5_D"][0].item()
                data_left["q6_B"] = info["q6_B"][0].item()
                data_left["l_4prime6"] = info["l_4prime6"][0].item()
                data_left["l_4prime7"] = info["l_4prime7"][0].item()
                data_left["l_57"] = info["l_57"][0].item()
                data_left["x_4prime6"] = info["x_4prime6"][0].item()
                data_left["x_4prime7"] = info["x_4prime7"][0].item()
                data_left["x_57"] = info["x_57"][0].item()
                data_left["phi_4prime_a"] = info["phi_4prime_a"][0].item()
                data_left["phi_4prime_b"] = info["phi_4prime_b"][0].item()
                data_left["phi_4prime_c"] = info["phi_4prime_c"][0].item()
                data_left["phi_4prime_d"] = info["phi_4prime_d"][0].item()
                data_left["phi_5_a"] = info["phi_5_a"][0].item()
                data_left["phi_5_b"] = info["phi_5_b"][0].item()
                data_left["h5_B"] = info["h5_B"][0].item()
                data_left["h5_C"] = info["h5_C"][0].item()
                data_left["h6_C"] = info["h6_C"][0].item()
                data_left["h6_D"] = info["h6_D"][0].item()
                data_left["tendon_torques"] = (
                    info["tendon_torques_left"][0].cpu().numpy().tolist()
                )

                data_right = {}
                data_right["state"] = ("s" if info["delta_l"][1].item() > 0 else "") + (
                    "a"
                    if info["a"][1].item() > 0
                    else (
                        "b"
                        if info["b"][1].item() > 0
                        else (
                            "c"
                            if info["c"][1].item() > 0
                            else ("d" if info["d"][1].item() > 0 else "x")
                        )
                    )
                )
                data_right["delta_l"] = info["delta_l"][1].item()
                data_right["thetas"] = info["thetas"][1].detach().cpu().numpy().tolist()
                data_right["qs"] = info["qs"][1].detach().cpu().numpy().tolist()
                data_right["q4"] = info["q4"][1].item()
                data_right["q4prime"] = info["q4prime"][1].item()
                data_right["q5_D"] = info["q5_D"][1].item()
                data_right["q6_B"] = info["q6_B"][1].item()
                data_right["l_4prime6"] = info["l_4prime6"][1].item()
                data_right["l_4prime7"] = info["l_4prime7"][1].item()
                data_right["l_57"] = info["l_57"][1].item()
                data_right["x_4prime6"] = info["x_4prime6"][1].item()
                data_right["x_4prime7"] = info["x_4prime7"][1].item()
                data_right["x_57"] = info["x_57"][1].item()
                data_right["phi_4prime_a"] = info["phi_4prime_a"][1].item()
                data_right["phi_4prime_b"] = info["phi_4prime_b"][1].item()
                data_right["phi_4prime_c"] = info["phi_4prime_c"][1].item()
                data_right["phi_4prime_d"] = info["phi_4prime_d"][1].item()
                data_right["phi_5_a"] = info["phi_5_a"][1].item()
                data_right["phi_5_b"] = info["phi_5_b"][1].item()
                data_right["h5_B"] = info["h5_B"][1].item()
                data_right["h5_C"] = info["h5_C"][1].item()
                data_right["h6_C"] = info["h6_C"][1].item()
                data_right["h6_D"] = info["h6_D"][1].item()
                data_right["tendon_torques"] = (
                    info["tendon_torques_right"][0].cpu().numpy().tolist()
                )
            else:
                gst_tendon_manager.apply_actuated_jit(**kwargs)

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

            # Record video frame if enabled
            if (
                args_cli.record_video
                and camera is not None
                and video_writer is not None
            ):
                camera.update(sim.get_physics_dt())
                # Get RGB image from camera (shape: [1, H, W, 3])
                rgb_image = camera.data.output["rgb"][0].cpu().numpy()
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_image)

            if not args_cli.jit:
                data_left["joint_pos"] = (
                    robot.data.joint_pos[0, joint_indices_left]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                data_right["joint_pos"] = (
                    robot.data.joint_pos[0, joint_indices_right]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                append_data(data_left, "left")
                append_data(data_right, "right")

    except KeyboardInterrupt as e:
        carb.log_error(f"An error occurred during simulation: {e}")
        # Release video writer on error
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {args_cli.video_output}")
        raise e

    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {args_cli.video_output}")

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
