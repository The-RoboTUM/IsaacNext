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
    "--controller",
    choices=("cpg", "sin"),
    default=None,
    help="Leg controller to use for actuated joints.",
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
args_cli.controller = args_cli.controller or FORREST_PARAMS.run.controller

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.tendons.controllers.base import DOF_ORDER, DOF_SIGN, DOF_TO_ACTUATOR_GROUP, LegControllerBase
from isaaclab.tendons.controllers.cpg import BirdBotCPGLeg, CPGParams
from isaaclab.tendons.controllers.sinusoidal import SinusoidalLegController, SinusoidalParams
from isaaclab.tendons.manager import TendonManager
from isaaclab.tendons.models.analytic.constants import joint_names_left, joint_names_right
from isaaclab.tendons.models.analytic.tendon_data import TendonData

from isaaclab_assets.robots.forrest import get_forrest_cfg

USD_PATH = "symlinks/forrest_urdf_latest/forrest_urdf_latest.usd"
VIRTUAL_GROUND_HEIGHT = FORREST_PARAMS.physics.virtual_ground_height
SIM_DT = FORREST_PARAMS.physics.sim_dt


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
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="world",
        ),
    )
    camera = TiledCamera(camera_cfg)

    video_output = Path(args.video_output)
    video_output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
    fps = 30  # int(1.0 / sim_cfg.dt)
    video_writer = cv2.VideoWriter(str(video_output), fourcc, fps, (1280, 720))
    return camera, video_writer


def add_fixed_world_joint(sim, params):
    """Lock /World/Bot/world_corrected to the world with a fixed USD joint."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = sim.stage
    body_path = Sdf.Path(params.robot.fixed_world_body_path)
    joint_path = Sdf.Path(params.robot.fixed_world_joint_path)
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"Cannot create fixed world joint: body prim does not exist: {body_path}")

    body_tf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    body_tf.Orthonormalize()
    body_pos_w = body_tf.ExtractTranslation()
    body_rot_w = body_tf.ExtractRotationQuat()

    if params.robot.fixed_world_joint_local_pos0 is None:
        local_pos0 = Gf.Vec3f(float(body_pos_w[0]), float(body_pos_w[1]), float(body_pos_w[2]))
    else:
        local_pos0 = Gf.Vec3f(*params.robot.fixed_world_joint_local_pos0)

    if params.robot.fixed_world_joint_local_rot0_wxyz is None:
        local_rot0 = Gf.Quatf(
            float(body_rot_w.real),
            float(body_rot_w.imaginary[0]),
            float(body_rot_w.imaginary[1]),
            float(body_rot_w.imaginary[2]),
        )
    else:
        local_rot0 = Gf.Quatf(*params.robot.fixed_world_joint_local_rot0_wxyz)

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed_joint.CreateBody1Rel().SetTargets([body_path])
    fixed_joint.CreateLocalPos0Attr(local_pos0)
    fixed_joint.CreateLocalRot0Attr(local_rot0)
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateCollisionEnabledAttr(False)


def make_actuated_dof_specs(robot_cfg):
    """Build target joint specs from the configured Forrest actuators.

    run.py intentionally does not contain concrete joint names. The actual
    joint regex/name strings come from the robot config actuator groups.
    """
    specs = []

    for side_prefix, side_name in (("l", "left"), ("r", "right")):
        for dof in DOF_ORDER:
            actuator_group = DOF_TO_ACTUATOR_GROUP[dof]
            actuator_cfg = robot_cfg.actuators[actuator_group]

            matches = [expr for expr in actuator_cfg.joint_names_expr if expr.startswith(side_prefix)]

            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {side_name} joint expression for "
                    f"controller DOF {dof!r} in actuator group {actuator_group!r}; "
                    f"got {matches}"
                )

            specs.append(
                {
                    "side": side_name,
                    "dof": dof,
                    "joint_expr": matches[0],
                    "sign": DOF_SIGN[dof],
                }
            )

    return specs


def find_actuated_joint_indices(robot, actuated_dof_specs):
    joint_exprs = [spec["joint_expr"] for spec in actuated_dof_specs]
    joint_indices, found_joint_names = robot.find_joints(joint_exprs, preserve_order=True)

    if len(joint_indices) != len(joint_exprs):
        raise RuntimeError(f"Could not find all actuated joints. Requested: {joint_exprs}; found: {found_joint_names}")

    print("Actuated controller DOFs:")
    for spec, joint_name in zip(actuated_dof_specs, found_joint_names):
        print(f"  {spec['side']:>5} {spec['dof']:<13} -> {joint_name}")

    return joint_indices


def controller_command_tensor(
    *,
    t: float,
    left_controller: LegControllerBase,
    right_controller: LegControllerBase,
    actuated_dof_specs,
    device,
) -> torch.Tensor:
    controllers = {
        "left": left_controller,
        "right": right_controller,
    }

    commands = []
    for spec in actuated_dof_specs:
        q, _qd = controllers[spec["side"]].joint(spec["dof"], t)
        commands.append(spec["sign"] * q)

    return torch.tensor([commands], dtype=torch.float32, device=device)


def make_cpg_legs() -> tuple[BirdBotCPGLeg, BirdBotCPGLeg]:
    """Create left/right CPG controllers."""
    phi_0_combined_offset = np.pi / 2

    common = dict(
        f_hz=1.5,
        D=0.60,
        A_h_deg=32.0,
        O_h_deg=22.0,
        A_k_deg=120.0,
        S_f=0.02,
        S_e=0.05,
    )
    left_params = CPGParams(phi0=-np.pi / 2 + phi_0_combined_offset, **common)
    right_params = CPGParams(phi0=np.pi / 2 + phi_0_combined_offset, **common)

    return (
        BirdBotCPGLeg(left_params, include_knee=True),
        BirdBotCPGLeg(right_params, include_knee=True),
    )


def make_sinusoidal_legs() -> tuple[SinusoidalLegController, SinusoidalLegController]:
    """Create left/right sinusoidal controllers over the same logical DOFs."""
    common = dict(
        f_hz=3.0,
        amplitude_deg={
            "hip_roll": 0.0,
            "hip_yaw": 0.0,
            "hip_flexion": 45.0,
            "knee_flexion": 75.0,
        },
        offset_deg={
            "hip_roll": 0.0,
            "hip_yaw": 0.0,
            "hip_flexion": 0.0,
            "knee_flexion": -75.0,
        },
    )

    return (
        SinusoidalLegController(
            SinusoidalParams(
                phi0=0.0,
                phase_rad={
                    "hip_flexion": 0.0,
                    "knee_flexion": 0.0,
                },
                **common,
            )
        ),
        SinusoidalLegController(
            SinusoidalParams(
                phi0=0.0,
                phase_rad={
                    "hip_flexion": 180.0,
                    "knee_flexion": 180.0,
                },
                **common,
            )
        ),
    )


def make_leg_controllers(controller_name: str) -> tuple[LegControllerBase, LegControllerBase]:
    if controller_name == "cpg":
        return make_cpg_legs()
    if controller_name == "sin":
        return make_sinusoidal_legs()
    raise ValueError(f"Unknown controller: {controller_name}")


def print_startup_summary(args, sim_cfg, num_steps: int):
    mode = "JIT / TorchScript" if args.jit else "DEBUG / eager"
    print("\n=== Forrest tendon simulation ===")
    print(f"Mode:              {mode}")
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
    num_steps = int(args_cli.duration / sim_cfg.dt)
    print_startup_summary(args_cli, sim_cfg, num_steps)

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

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

    sim.reset()
    add_fixed_world_joint(sim, FORREST_PARAMS)
    # robot.write_joint_state_to_sim(
    #     position=robot.data.default_joint_pos,
    #     velocity=robot.data.default_joint_vel,
    # )
    # robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    time.sleep(0.1)

    joint_indices_right, _ = robot.find_joints(joint_names_right, preserve_order=True)
    joint_indices_left, _ = robot.find_joints(joint_names_left, preserve_order=True)
    actuated_dof_specs = make_actuated_dof_specs(robot_cfg)
    actuated_joint_indices = find_actuated_joint_indices(robot, actuated_dof_specs)

    tendon_manager = TendonManager(
        robot,
        tendon_data=TendonData(
            robot.num_instances,
            FORREST_PARAMS.to_tendon_randomization_ranges(),
            tc=FORREST_PARAMS.to_tendon_constants(),
        ),
        tendon_damping=FORREST_PARAMS.tendon_damping(),
    )
    left_controller, right_controller = make_leg_controllers(args_cli.controller)

    mode_label = "jit" if args_cli.jit else "debug"
    wall_start = time.perf_counter()

    try:
        for iteration in range(num_steps):
            t = iteration * sim.get_physics_dt()
            debug_info = None

            if args_cli.jit:
                tendon_manager.apply_jit(virtual_ground_height=VIRTUAL_GROUND_HEIGHT, dt=SIM_DT)
            else:
                debug_info = tendon_manager.apply_debug(virtual_ground_height=VIRTUAL_GROUND_HEIGHT, dt=SIM_DT)
                data_left, data_right = leg_tensordict_to_python_dict(debug_info)

            commanded_positions = controller_command_tensor(
                t=t,
                left_controller=left_controller,
                right_controller=right_controller,
                actuated_dof_specs=actuated_dof_specs,
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
                camera.update(sim.get_physics_dt())
                rgb_image = camera.data.output["rgb"][0].cpu().numpy()
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_image)

            if debug_info is not None:
                data_left["joint_pos"] = tensor_to_python(robot.data.joint_pos[0, joint_indices_left])
                data_right["joint_pos"] = tensor_to_python(robot.data.joint_pos[0, joint_indices_right])
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
