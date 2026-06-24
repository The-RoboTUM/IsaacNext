# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Forrest tendon simulation in either debug or TorchScript/JIT mode.

Debug mode keeps rich tendon diagnostics and writes JSONL logs.
JIT mode runs the fast tensor-only tendon path and prints lightweight progress.
"""

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/linus/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/cudnn/lib

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher
from isaaclab.tendons.parameter_loader import load_forrest_parameter_config

parser = argparse.ArgumentParser(description="Run the Forrest tendon simulation.")
parser.add_argument("--jit", action="store_true", help="Use the TorchScript/JIT tendon path.")
parser.add_argument("--record_video", action="store_true", help="Record video of the simulation.")
parser.add_argument(
    "--video_output",
    type=str,
    default=None,
    help="Output path for the video.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory used for debug JSONL logs and optional video output.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=None,
    help="Simulation duration in seconds.",
)
parser.add_argument(
    "--status_interval",
    type=int,
    default=None,
    help="Print one status line every N simulation steps.",
)
parser.add_argument(
    "--startup_hold",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Hold the measured initial joint targets before starting the gait controller.",
)
parser.add_argument(
    "--startup_hold_duration",
    type=float,
    default=None,
    help="Startup hold duration in seconds.",
)
parser.add_argument(
    "--controller",
    choices=("cpg", "cpg_oscillator", "sin"),
    default=None,
    help="Leg controller to use for actuated joints.",
)
parser.add_argument(
    "--constraint_mode",
    choices=("freefall", "boom", "static"),
    default=None,
    help=(
        "Base constraint mode: freefall creates no world constraint, boom locks motion with the configured sagittal "
        "plane D6 joint, static creates the configured fixed-world joint."
    ),
)
parser.add_argument(
    "--parameters_file",
    type=str,
    default=None,
    help="Path to a Forrest parameter YAML file or profile directory.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

FORREST_PARAMS = load_forrest_parameter_config(args_cli.parameters_file)
args_cli.video_output = args_cli.video_output or FORREST_PARAMS.run.video_output
args_cli.output_dir = args_cli.output_dir or FORREST_PARAMS.run.output_dir
args_cli.duration = args_cli.duration if args_cli.duration is not None else FORREST_PARAMS.run.duration
args_cli.status_interval = (
    args_cli.status_interval if args_cli.status_interval is not None else FORREST_PARAMS.run.status_interval
)
args_cli.startup_hold = (
    args_cli.startup_hold if args_cli.startup_hold is not None else FORREST_PARAMS.run.startup_hold_enabled
)
args_cli.startup_hold_duration = (
    args_cli.startup_hold_duration
    if args_cli.startup_hold_duration is not None
    else FORREST_PARAMS.run.startup_hold_duration
)
args_cli.controller = args_cli.controller or FORREST_PARAMS.run.controller
args_cli.constraint_mode = args_cli.constraint_mode or FORREST_PARAMS.run.constraint_mode

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.tendons.manager import TendonManager
from isaaclab.tendons.models.analytic.constants import joint_names_left, joint_names_right
from isaaclab.tendons.models.analytic.tendon_data import TendonData
from isaaclab.tendons.runner import (
    configure_base_constraint,
    controller_command_tensor,
    find_actuated_joint_indices,
    make_actuated_dof_specs,
    make_leg_controllers,
    reset_robot_to_default,
)
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from isaaclab_assets.robots.forrest import get_forrest_cfg

USD_PATH = "symlinks/forrest_ws/urdf/forrest_isaac/forrest_isaac.usd"
VIRTUAL_GROUND_HEIGHT = FORREST_PARAMS.physics.virtual_ground_height
SIM_DT = FORREST_PARAMS.physics.sim_dt
CAMERA_EYE = (2.5, -8.0, 2.0)
CAMERA_TARGET = (2.5, 0.0, 0.85)
_CV2 = None


# Enable this while hunting for autograd issues in the tendon model.
# torch.autograd.set_detect_anomaly(True)


def tensor_to_python(value: torch.Tensor):
    """Detach a tensor and convert it to a JSON-serializable Python value."""
    value = value.detach().cpu()
    if value.ndim == 0:
        return value.item()
    return value.numpy().tolist()


def leg_tensordict_to_python_dict(tensordict):
    """Split a two-leg tendon debug tensor dictionary into left/right JSON dictionaries."""
    data_left = {}
    data_right = {}

    for key, value in tensordict.items():
        if key == "tendon_torques_left":
            data_left["tendon_torques"] = tensor_to_python(value)
            continue
        if key == "tendon_torques_right":
            data_right["tendon_torques"] = tensor_to_python(value)
            continue

        value_left = value[0]
        value_right = value[1]
        if value_left.ndim <= 1:
            data_left[key] = tensor_to_python(value_left)
            data_right[key] = tensor_to_python(value_right)
        else:
            raise ValueError(f"Unsupported value shape {value_left.shape} for key {key}")

    return data_left, data_right


def append_jsonl(path: Path, data: dict):
    """Append one JSON object to a JSONL file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data) + "\n")


def require_cv2():
    """Import OpenCV only when video recording is requested."""
    global _CV2
    if _CV2 is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires OpenCV. Install the repo environment from environment.yml "
                "or add opencv-python to your active Python environment."
            ) from exc
        _CV2 = cv2_module
    return _CV2


def reset_debug_logs(output_dir: Path) -> tuple[Path, Path]:
    """Create empty debug log files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    left_path = output_dir / "gst_data_left.jsonl"
    right_path = output_dir / "gst_data_right.jsonl"
    left_path.write_text("", encoding="utf-8")
    right_path.write_text("", encoding="utf-8")
    return left_path, right_path


def setup_video_writer(args, sim_cfg):
    """Create the camera and OpenCV writer used for optional recording."""
    if not args.record_video:
        return None, None

    cv2 = require_cv2()
    import isaacsim.core.utils.prims as prim_utils

    prim_utils.create_prim("/World/Camera", "Xform")
    camera_eye = torch.tensor([CAMERA_EYE], dtype=torch.float32)
    camera_target = torch.tensor([CAMERA_TARGET], dtype=torch.float32)
    camera_rot = quat_from_matrix(create_rotation_matrix_from_view(camera_eye, camera_target, up_axis="Z"))[0]

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
            pos=CAMERA_EYE,
            rot=tuple(float(value) for value in camera_rot),
            convention="opengl",
        ),
    )
    camera = TiledCamera(camera_cfg)

    video_output = Path(args.video_output)
    video_output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
    fps = 30  # int(1.0 / sim_cfg.dt)
    video_writer = cv2.VideoWriter(str(video_output), fourcc, fps, (1280, 720))
    return camera, video_writer


def print_startup_summary(args, sim_cfg, num_steps: int):
    mode = "JIT / TorchScript" if args.jit else "DEBUG / eager"
    startup_hold_duration = args.startup_hold_duration if args.startup_hold else 0.0
    print("\n=== Forrest tendon simulation ===")
    print(f"Mode:              {mode}")
    print(f"Base constraint:   {args.constraint_mode}")
    print(f"Controller:        {args.controller}")
    print(f"Startup hold:      {startup_hold_duration:.3f} s")
    print(f"Isaac device:      {args.device}")
    print(f"Torch CUDA:        {torch.cuda.is_available()}")
    print(f"Physics dt:        {sim_cfg.dt:.6f} s")
    print(f"Duration:          {args.duration:.3f} s ({num_steps} steps)")
    virtual_ground_str = "disabled" if VIRTUAL_GROUND_HEIGHT is None else f"{VIRTUAL_GROUND_HEIGHT:.3f} m"

    print(f"Virtual ground:    {virtual_ground_str}")
    print(f"Video recording:   {'on' if args.record_video else 'off'}")

    if args.jit:
        print("Diagnostics:       lightweight console status only")
    else:
        print(f"Diagnostics:       debug JSONL logs in {args.output_dir}/")
    print("=================================\n")


def maybe_print_status(
    *,
    iteration: int,
    num_steps: int,
    sim_time: float,
    wall_start: float,
    status_interval: int,
    mode: str,
    debug_info: dict | None = None,
):
    """Print a compact progress line in both debug and JIT mode."""
    if status_interval <= 0:
        return
    if iteration % status_interval != 0 and iteration != num_steps - 1:
        return

    elapsed = max(time.perf_counter() - wall_start, 1.0e-9)
    steps_per_sec = (iteration + 1) / elapsed
    prefix = f"[{mode}] step {iteration + 1:5d}/{num_steps} | t={sim_time:6.3f}s | {steps_per_sec:7.1f} steps/s"

    if debug_info is None:
        print(prefix)
        return

    torque_stats = []
    for key in ("tendon_torques_left", "tendon_torques_right"):
        value = debug_info.get(key)
        if isinstance(value, torch.Tensor):
            torque_stats.append(
                f"{key.replace('tendon_torques_', '')}_|tau|max={value.detach().abs().max().item():.3e}"
            )

    if torque_stats:
        print(prefix + " | " + " | ".join(torque_stats))
    else:
        print(prefix)


def main():
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=tuple(FORREST_PARAMS.physics.gravity))
    sim_cfg.dt = SIM_DT
    sim_cfg.physx.enable_external_forces_every_iteration = True
    sim_cfg.physx.min_velocity_iteration_count = max(1, int(sim_cfg.physx.min_velocity_iteration_count))
    sim_cfg.physx.gpu_collision_stack_size = int(FORREST_PARAMS.physics.physx_gpu_collision_stack_size)
    sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = int(
        FORREST_PARAMS.physics.physx_gpu_found_lost_aggregate_pairs_capacity
    )
    sim_cfg.physx.gpu_total_aggregate_pairs_capacity = int(
        FORREST_PARAMS.physics.physx_gpu_total_aggregate_pairs_capacity
    )
    num_steps = max(1, int(args_cli.duration / sim_cfg.dt))
    print_startup_summary(args_cli, sim_cfg, num_steps)

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

    robot_cfg = get_forrest_cfg(FORREST_PARAMS).replace(prim_path="/World/Bot")
    robot = Articulation(robot_cfg)

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DomeLightCfg())

    debug_left_path = None
    debug_right_path = None
    if not args_cli.jit:
        debug_left_path, debug_right_path = reset_debug_logs(output_dir)
        print(f"Debug logs reset: {debug_left_path}, {debug_right_path}")

    camera, video_writer = setup_video_writer(args_cli, sim_cfg)

    configure_base_constraint(sim, FORREST_PARAMS, args_cli.constraint_mode)
    sim.reset()
    reset_robot_to_default(robot)
    robot.update(0.0)

    joint_indices_right, _ = robot.find_joints(joint_names_right, preserve_order=True)
    joint_indices_left, _ = robot.find_joints(joint_names_left, preserve_order=True)

    sim.step()
    robot.update(sim.get_physics_dt())
    time.sleep(0.1)

    actuated_dof_specs = make_actuated_dof_specs(robot_cfg)
    actuated_joint_indices = find_actuated_joint_indices(robot, actuated_dof_specs)
    FORREST_PARAMS.run.controller = args_cli.controller
    left_controller, right_controller = make_leg_controllers(FORREST_PARAMS.run)
    initial_controller_targets = controller_command_tensor(
        t=0.0,
        left_controller=left_controller,
        right_controller=right_controller,
        actuated_dof_specs=actuated_dof_specs,
        initial_joint_positions=torch.zeros((1, len(actuated_joint_indices)), dtype=torch.float32, device=robot.device),
        controller_zero=torch.zeros((1, len(actuated_joint_indices)), dtype=torch.float32, device=robot.device),
        device=robot.device,
    )
    initial_joint_positions = robot.data.joint_pos[:, actuated_joint_indices].clone()
    startup_hold_duration = max(0.0, float(args_cli.startup_hold_duration)) if args_cli.startup_hold else 0.0

    tendon_manager = TendonManager(
        robot,
        tendon_data=TendonData(
            robot.num_instances,
            FORREST_PARAMS.to_tendon_randomization_ranges(),
            tc=FORREST_PARAMS.to_tendon_constants(device=robot.device),
            device=robot.device,
        ),
        tendon_damping=FORREST_PARAMS.tendon_damping(),
    )

    mode_label = "jit" if args_cli.jit else "debug"
    wall_start = time.perf_counter()

    try:
        for iteration in range(num_steps):
            t = iteration * sim.get_physics_dt()
            debug_info = None
            debug_joint_pos_left = None
            debug_joint_pos_right = None

            if args_cli.jit:
                tendon_manager.apply_jit(virtual_ground_height=VIRTUAL_GROUND_HEIGHT, dt=SIM_DT)
            else:
                # apply_debug reads the current pre-step joint state; keep the JSONL joint_pos aligned with it.
                debug_joint_pos_left = robot.data.joint_pos[0, joint_indices_left].detach().clone()
                debug_joint_pos_right = robot.data.joint_pos[0, joint_indices_right].detach().clone()
                debug_info = tendon_manager.apply_debug(virtual_ground_height=VIRTUAL_GROUND_HEIGHT, dt=SIM_DT)
                data_left, data_right = leg_tensordict_to_python_dict(debug_info)

            controller_t = max(0.0, t - startup_hold_duration)
            if t < startup_hold_duration:
                commanded_positions = initial_joint_positions
            else:
                commanded_positions = controller_command_tensor(
                    t=controller_t,
                    left_controller=left_controller,
                    right_controller=right_controller,
                    actuated_dof_specs=actuated_dof_specs,
                    initial_joint_positions=initial_joint_positions,
                    controller_zero=initial_controller_targets,
                    device=robot.device,
                )

            robot.set_joint_position_target(
                commanded_positions,
                joint_ids=actuated_joint_indices,
            )

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

            if args_cli.record_video and camera is not None and video_writer is not None:
                cv2 = require_cv2()
                camera.update(sim.get_physics_dt())
                rgb_image = camera.data.output["rgb"][0].cpu().numpy()
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_image)

            if debug_info is not None:
                data_left["sim_time"] = t
                data_right["sim_time"] = t
                data_left["joint_pos"] = tensor_to_python(debug_joint_pos_left)
                data_right["joint_pos"] = tensor_to_python(debug_joint_pos_right)
                data_left["joint_pos_after_step"] = tensor_to_python(robot.data.joint_pos[0, joint_indices_left])
                data_right["joint_pos_after_step"] = tensor_to_python(robot.data.joint_pos[0, joint_indices_right])
                append_jsonl(debug_left_path, data_left)
                append_jsonl(debug_right_path, data_right)

            maybe_print_status(
                iteration=iteration,
                num_steps=num_steps,
                sim_time=t,
                wall_start=wall_start,
                status_interval=args_cli.status_interval,
                mode=mode_label,
                debug_info=debug_info,
            )

    except KeyboardInterrupt as exc:
        carb.log_error(f"Simulation interrupted: {exc}")
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {args_cli.video_output}")

    print("\nCompleted simulation.")
    if not args_cli.jit:
        print(f"Debug logs written to: {debug_left_path} and {debug_right_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
