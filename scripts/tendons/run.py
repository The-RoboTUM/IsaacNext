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
from datetime import datetime
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
    "--num_envs",
    type=int,
    default=None,
    help="Number of parallel simulation environments to create for recording.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Spacing between cloned environments.",
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
    choices=("freefall", "boom", "static", "static_boom"),
    default=None,
    help=(
        "Base constraint mode: freefall creates no world constraint, boom locks motion with the configured sagittal "
        "plane D6 joint, static creates the configured fixed-world joint, static_boom creates both."
    ),
)
parser.add_argument(
    "--parameters_file",
    type=str,
    default=None,
    help="Path to a Forrest parameter YAML file or profile directory.",
)
parser.add_argument(
    "--calibration",
    action="store_true",
    help="Open live calibration controls and plot windows.",
)
parser.add_argument("--record_identix", action="store_true", help="Record Identix-compatible sim_data.")
parser.add_argument(
    "--record_output_dir",
    type=str,
    default=None,
    help=(
        "Exact output directory for the Identix SQLite database and metadata. "
        "If omitted, uses outputs/forrest_dbs_<timestamp>."
    ),
)
parser.add_argument(
    "--record_side",
    choices=("left", "right", "both", "full"),
    default=None,
    help="Leg side to record. 'both' stores each side as separate samples; 'full' stores base plus both legs.",
)
parser.add_argument(
    "--record_joint_set",
    choices=("real_leg_joints", "tendon_chain_5"),
    default=None,
    help="Joint set to store in sim_data.",
)
parser.add_argument(
    "--record_spatial_state",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Deprecated for Identix kinematics DB output; spatial tables are not written.",
)
parser.add_argument(
    "--record_body_set",
    choices=("tendon_chain_links",),
    default=None,
    help="Body/link set to store in spatial diagnostics.",
)
parser.add_argument(
    "--record_tau_source",
    choices=(
        "actuation_command",
        "motor_torque",
        "controller_plus_ground",
        "applied_torque",
        "computed_torque",
        "zero",
    ),
    default=None,
    help="Torque tensor used for tau0..tauN in sim_data.",
)
parser.add_argument("--record_stride", type=int, default=None, help="Record every N simulation steps.")
parser.add_argument(
    "--record_start_time",
    type=float,
    default=None,
    help="Skip recording until this simulation time in seconds.",
)
parser.add_argument(
    "--record_tendons",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Write forrest_tendons.db with visualization/debug tendon frames when debug data is available.",
)
parser.add_argument(
    "--record_dynamics",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Write forrest_dynamics.db with non-tendon inverse-dynamics terms aligned to sim_data.",
)
parser.add_argument(
    "--record_overwrite",
    action="store_true",
    help="Overwrite an existing Identix recording in the output directory.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
CONSTRAINED_BASE_MODES = ("boom", "static", "static_boom")

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
args_cli.record_identix = bool(args_cli.record_identix or FORREST_PARAMS.recording.enabled)
args_cli.record_side = args_cli.record_side or FORREST_PARAMS.recording.side
args_cli.record_joint_set = args_cli.record_joint_set or FORREST_PARAMS.recording.joint_set
args_cli.record_spatial_state = (
    args_cli.record_spatial_state
    if args_cli.record_spatial_state is not None
    else FORREST_PARAMS.recording.record_spatial_state
)
args_cli.record_body_set = args_cli.record_body_set or FORREST_PARAMS.recording.body_set
args_cli.record_tau_source = args_cli.record_tau_source or FORREST_PARAMS.recording.tau_source
args_cli.record_stride = (
    args_cli.record_stride if args_cli.record_stride is not None else FORREST_PARAMS.recording.stride
)
args_cli.record_start_time = (
    args_cli.record_start_time if args_cli.record_start_time is not None else FORREST_PARAMS.recording.start_time
)
args_cli.record_tendons = (
    args_cli.record_tendons if args_cli.record_tendons is not None else FORREST_PARAMS.recording.record_tendons
)
args_cli.record_dynamics = (
    args_cli.record_dynamics if args_cli.record_dynamics is not None else FORREST_PARAMS.recording.record_dynamics
)
args_cli.record_overwrite = bool(args_cli.record_overwrite or FORREST_PARAMS.recording.overwrite)
recording_base = bool(FORREST_PARAMS.recording.record_base_state or args_cli.record_side == "full")
record_stabilization_contact = bool(recording_base and args_cli.constraint_mode in CONSTRAINED_BASE_MODES)
if args_cli.record_output_dir is None:
    args_cli.record_output_dir = FORREST_PARAMS.recording.output_dir
if args_cli.record_output_dir is None:
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args_cli.record_output_dir = str(Path(args_cli.output_dir) / f"forrest_dbs_{datestamp}")
if args_cli.record_identix:
    physx_tensor_log_filter = "--/log/channels/omni.physx.tensors.plugin=error"
    kit_args = getattr(args_cli, "kit_args", "") or ""
    if "--/log/channels/omni.physx.tensors.plugin=" not in kit_args:
        args_cli.kit_args = f"{kit_args} {physx_tensor_log_filter}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.tendons.data_recording import (
    DataRecording,
    DataRecordingConfig,
    actuation_command_tensor,
    motor_torque_tensor,
)
from isaaclab.tendons.manager import TendonManager
from isaaclab.tendons.models.analytic.constants import joint_names_left, joint_names_right
from isaaclab.tendons.models.analytic.tendon_data import TendonData
from isaaclab.tendons.rl_recording import (
    _dynamics_terms as shared_recording_dynamics_terms,
)
from isaaclab.tendons.rl_recording import (
    _joint_to_full_generalized as shared_joint_to_full_generalized,
)
from isaaclab.tendons.rl_recording import (
    _projected_contact_sensor_torque as shared_projected_contact_sensor_torque,
)
from isaaclab.tendons.runner import (
    configure_scene_base_constraints,
    controller_command_tensor,
    find_actuated_joint_indices,
    make_actuated_dof_specs,
    make_leg_controllers,
    reset_robot_to_default,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_apply, quat_from_matrix

CONTACT_GROUP_PATTERNS = {
    "digit": ("digit",),
    "connector": ("foot_connector",),
    "base": ("base", "hip", "differential_cage"),
    "self_collision": ("s23",),
}

if args_cli.calibration:
    from isaaclab.tendons.calibration import (
        CalibrationWindows,
        ForrestTendonOverlay,
        apply_tendon_parameters,
        build_calibration_state,
        build_tendon_data_from_state,
        runtime_controller_command_tensor,
    )

from isaaclab_assets.robots.forrest import get_forrest_cfg

USD_PATH = "symlinks/forrest_ws/urdf/forrest_isaac/forrest_isaac.usd"
SIM_DT = FORREST_PARAMS.physics.sim_dt
VIRTUAL_GROUND_HEIGHT = None
CAMERA_EYE = (2.5, -8.0, 2.0)
CAMERA_TARGET = (2.5, 0.0, 0.85)
_CV2 = None


def body_name_regex(body_names: tuple[str, ...] | list[str]) -> str:
    escaped = [name.replace(".", r"\.") for name in body_names]
    return "(" + "|".join(escaped) + ")"


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


def make_ground_material_cfg():
    material = FORREST_PARAMS.physics.ground
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=material.static_friction,
        dynamic_friction=material.dynamic_friction,
        restitution=material.restitution,
        friction_combine_mode=material.friction_combine_mode,
        restitution_combine_mode=material.restitution_combine_mode,
    )


def make_foot_material_cfg():
    events = FORREST_PARAMS.training.events
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=events.foot_static_friction_range[0],
        dynamic_friction=events.foot_dynamic_friction_range[0],
        restitution=events.foot_restitution_range[0],
        friction_combine_mode=FORREST_PARAMS.physics.ground.friction_combine_mode,
        restitution_combine_mode=FORREST_PARAMS.physics.ground.restitution_combine_mode,
    )


def apply_foot_material(robot_prim_path: str) -> None:
    material_path = f"{robot_prim_path}/rubberFootPhysicsMaterial"
    material_cfg = make_foot_material_cfg()
    material_cfg.func(material_path, material_cfg)
    for body_name in FORREST_PARAMS.training.contacts.foot_body_names:
        body_path = f"{robot_prim_path}/{body_name}"
        try:
            bound = sim_utils.bind_physics_material(body_path, material_path)
        except ValueError as exc:
            carb.log_warn(f"Unable to bind foot physics material to {body_path}: {exc}")
            continue
        if not bound:
            carb.log_warn(f"Unable to bind foot physics material to {body_path}")
        else:
            carb.log_info(f"Bound rubber foot physics material to {body_path}")


def make_env_time_offsets(*, controller: str, num_envs: int, device) -> torch.Tensor:
    if num_envs <= 1:
        return torch.zeros(1, device=device, dtype=torch.float32)

    if controller == "cpg":
        frequency_hz = float(FORREST_PARAMS.run.cpg.f_hz)
    elif controller == "cpg_oscillator":
        frequency_hz = float(FORREST_PARAMS.run.cpg_oscillator.f_hz)
    else:
        frequency_hz = float(FORREST_PARAMS.run.sinusoidal.f_hz)

    cycle_period = 1.0 / max(frequency_hz, 1.0e-6)
    return torch.arange(num_envs, device=device, dtype=torch.float32) * (cycle_period / float(num_envs))


def batched_controller_command_tensor(
    *,
    t: float,
    env_time_offsets: torch.Tensor,
    left_controller,
    right_controller,
    actuated_dof_specs,
    initial_joint_positions: torch.Tensor,
    device,
) -> torch.Tensor:
    commanded_positions = torch.empty_like(initial_joint_positions)
    for env_id in range(initial_joint_positions.shape[0]):
        env_command = controller_command_tensor(
            t=t + float(env_time_offsets[env_id].item()),
            left_controller=left_controller,
            right_controller=right_controller,
            actuated_dof_specs=actuated_dof_specs,
            initial_joint_positions=initial_joint_positions[env_id : env_id + 1],
            device=device,
        )
        commanded_positions[env_id] = env_command[0]
    return commanded_positions


def batched_runtime_controller_command_tensor(
    *,
    t: float,
    env_time_offsets: torch.Tensor,
    state,
    actuated_dof_specs,
    initial_joint_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    commanded_positions = torch.empty_like(initial_joint_positions)
    controller_delta = torch.empty_like(initial_joint_positions)
    for env_id in range(initial_joint_positions.shape[0]):
        env_command, env_delta = runtime_controller_command_tensor(
            t=t + float(env_time_offsets[env_id].item()),
            state=state,
            actuated_dof_specs=actuated_dof_specs,
            initial_joint_positions=initial_joint_positions[env_id : env_id + 1],
        )
        commanded_positions[env_id] = env_command[0]
        controller_delta[env_id] = env_delta[0]
    return commanded_positions, controller_delta


def make_forrest_recording_scene_cfg(*, num_envs: int, env_spacing: float):
    ground_cfg = sim_utils.GroundPlaneCfg(physics_material=make_ground_material_cfg())
    contact_body_names = FORREST_PARAMS.training.contacts.contact_sensor_body_names
    contact_filter_paths = ["/World/defaultGroundPlane/GroundPlane/CollisionPlane"]
    contact_filter_paths.extend(f"{FORREST_PARAMS.robot.prim_path}/{body_name}" for body_name in contact_body_names)

    @configclass
    class ForrestRecordingSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=ground_cfg)
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot = get_forrest_cfg(FORREST_PARAMS).replace(prim_path=FORREST_PARAMS.robot.prim_path)
        contact_forces = ContactSensorCfg(
            prim_path=f"{FORREST_PARAMS.robot.prim_path}/{body_name_regex(contact_body_names)}",
            update_period=SIM_DT,
            history_length=1,
            track_air_time=False,
            track_contact_points=True,
            track_friction_forces=True,
            max_contact_data_count_per_prim=256,
            filter_prim_paths_expr=contact_filter_paths,
        )

    replicate_physics = num_envs <= 1 or args_cli.constraint_mode == "freefall"
    if num_envs > 1 and not replicate_physics:
        print("[ForrestRun] Disabling replicated physics for constrained parallel environments.")
    return ForrestRecordingSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=replicate_physics,
    )


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


def print_startup_summary(args, sim_cfg, num_steps: int, *, num_envs: int, env_spacing: float):
    mode = "JIT / TorchScript" if args.jit else "DEBUG / eager"
    startup_hold_duration = args.startup_hold_duration if args.startup_hold else 0.0
    ground_material = FORREST_PARAMS.physics.ground
    event_params = FORREST_PARAMS.training.events
    print("\n=== Forrest tendon simulation ===")
    print(f"Mode:              {mode}")
    print(f"Base constraint:   {args.constraint_mode}")
    print(f"Controller:        {args.controller}")
    print(f"Startup hold:      {startup_hold_duration:.3f} s")
    print(f"Isaac device:      {args.device}")
    print(f"Torch CUDA:        {torch.cuda.is_available()}")
    print(f"Physics dt:        {sim_cfg.dt:.6f} s")
    print(f"Duration:          {args.duration:.3f} s ({num_steps} steps)")
    print(f"Parallel envs:     {num_envs}")
    print(f"Env spacing:       {env_spacing:.3f} m")
    virtual_ground_str = "disabled" if VIRTUAL_GROUND_HEIGHT is None else f"{VIRTUAL_GROUND_HEIGHT:.3f} m"

    print(f"Virtual ground:    {virtual_ground_str}")
    print(
        "Ground friction:   "
        f"static={ground_material.static_friction:.3f}, "
        f"dynamic={ground_material.dynamic_friction:.3f}, "
        f"combine={ground_material.friction_combine_mode}"
    )
    print(
        "Foot friction:     "
        f"static={event_params.foot_static_friction_range[0]:.3f}, "
        f"dynamic={event_params.foot_dynamic_friction_range[0]:.3f}"
    )
    print(f"Video recording:   {'on' if args.record_video else 'off'}")
    print(f"Identix recording: {'on' if args.record_identix else 'off'}")
    if args.record_identix:
        print(f"Recording output:  {args.record_output_dir}")
        print(f"Recording side:    {args.record_side}")
        print(f"Recording base:    {'on' if recording_base else 'off'}")
        print(f"Stab as contact:   {'on' if record_stabilization_contact else 'off'}")
        print(f"Recording stride:  {args.record_stride}")
        print(f"Recording tau:     {args.record_tau_source}")
        print(f"Recording ddq:     {FORREST_PARAMS.recording.ddq_source}")
        print(f"Recording tendons: {'on' if args.record_tendons and not args.jit else 'off'}")
        print(f"Recording dynamics:{'on' if args.record_dynamics else 'off'}")
    print(f"Calibration UI:    {'on' if args.calibration else 'off'}")

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


def projected_contact_sensor_torque_parts(robot, contact_sensor: ContactSensor, contact_body_names: tuple[str, ...]):
    """Project measured world-frame contact wrench into generalized joint torques."""

    num_joints = robot.data.joint_pos.shape[1]
    tau_force = torch.zeros_like(robot.data.joint_pos)
    tau_moment = torch.zeros_like(robot.data.joint_pos)
    available_names = set(contact_sensor.body_names)
    selected_names = tuple(name for name in contact_body_names if name in available_names)
    if not selected_names:
        return tau_force, tau_moment

    sensor_body_indices = [contact_sensor.body_names.index(name) for name in selected_names]
    robot_body_indices, _ = robot.find_bodies(list(selected_names), preserve_order=True)

    if contact_sensor.data.force_matrix_w is not None:
        normal_forces_by_filter = contact_sensor.data.force_matrix_w[:, sensor_body_indices, :, :]
    else:
        normal_forces_by_filter = contact_sensor.data.net_forces_w[:, sensor_body_indices, :].unsqueeze(2)
    friction_forces = getattr(contact_sensor.data, "friction_forces_w", None)
    if friction_forces is not None:
        friction_forces_by_filter = friction_forces[:, sensor_body_indices, :, :]
    else:
        friction_forces_by_filter = torch.zeros_like(normal_forces_by_filter)
    forces_by_filter = normal_forces_by_filter + friction_forces_by_filter
    forces_world = forces_by_filter.sum(dim=2)
    if not torch.any(forces_world).item():
        return tau_force, tau_moment

    contact_pos = getattr(contact_sensor.data, "contact_pos_w", None)
    if contact_pos is not None:
        selected_contact_pos = contact_pos[:, sensor_body_indices, :, :]
        valid_contact = torch.isfinite(selected_contact_pos).all(dim=-1)
        force_weights = torch.linalg.norm(forces_by_filter, dim=-1) * valid_contact.to(dtype=forces_by_filter.dtype)
        weight_sum = force_weights.sum(dim=2, keepdim=True).clamp_min(1.0e-12)
        contact_pos_world = (torch.nan_to_num(selected_contact_pos, nan=0.0) * force_weights.unsqueeze(-1)).sum(
            dim=2
        ) / weight_sum
    else:
        contact_pos_world = None

    joint_ids = list(range(num_joints))
    jacobian_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    jacobians = robot.root_physx_view.get_jacobians()

    for local_foot_index, body_index in enumerate(robot_body_indices):
        jacobian_body_index = int(body_index) - 1 if robot.is_fixed_base else int(body_index)
        jacobian_linear = jacobians[:, jacobian_body_index, 0:3, :][:, :, jacobian_joint_ids]
        jacobian_angular = jacobians[:, jacobian_body_index, 3:6, :][:, :, jacobian_joint_ids]
        force = forces_world[:, local_foot_index, :]
        body_pos = robot.data.body_pos_w[:, int(body_index), :]
        if contact_pos_world is not None:
            # Match the contact sensor force sign to the angular Jacobian wrench convention.
            moment = torch.cross(force, contact_pos_world[:, local_foot_index, :] - body_pos, dim=1)
        else:
            moment = torch.zeros_like(force)
        tau_force += torch.bmm(jacobian_linear.transpose(1, 2), force.unsqueeze(-1)).squeeze(-1)
        tau_moment += torch.bmm(jacobian_angular.transpose(1, 2), moment.unsqueeze(-1)).squeeze(-1)

    return tau_force, tau_moment


def projected_contact_sensor_torque_groups(
    robot,
    contact_sensor: ContactSensor,
    contact_body_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    grouped: dict[str, torch.Tensor] = {}
    for group_name, patterns in CONTACT_GROUP_PATTERNS.items():
        names = tuple(name for name in contact_body_names if any(pattern in name for pattern in patterns))
        tau_force, tau_moment = projected_contact_sensor_torque_parts(robot, contact_sensor, names)
        grouped[f"contact_{group_name}_force"] = tau_force
        grouped[f"contact_{group_name}_moment"] = tau_moment
        grouped[f"contact_{group_name}"] = tau_force + tau_moment
    return grouped


def zero_contact_sensor_torque_groups(robot) -> dict[str, torch.Tensor]:
    grouped: dict[str, torch.Tensor] = {}
    for group_name in CONTACT_GROUP_PATTERNS:
        zero = torch.zeros_like(robot.data.joint_pos)
        grouped[f"contact_{group_name}_force"] = zero
        grouped[f"contact_{group_name}_moment"] = zero
        grouped[f"contact_{group_name}"] = zero
    return grouped


def projected_contact_sensor_torque(robot, contact_sensor: ContactSensor, contact_body_names: tuple[str, ...]):
    tau_force, tau_moment = projected_contact_sensor_torque_parts(robot, contact_sensor, contact_body_names)
    return tau_force + tau_moment


def base_constraint_contact_tensor(robot, constraint_mode: str):
    if constraint_mode not in CONSTRAINED_BASE_MODES:
        return shared_joint_to_full_generalized(robot, robot.data.joint_pos * 0.0)

    base_contact = shared_joint_to_full_generalized(robot, robot.data.joint_pos * 0.0)
    wrench_width = robot.data.body_incoming_joint_wrench_b.shape[-1]
    base_contact[:, :wrench_width] = robot.data.body_incoming_joint_wrench_b[:, 0, :]
    return base_contact


def recording_tau_tensor(
    robot,
    contact_sensor: ContactSensor,
    contact_body_names: tuple[str, ...],
    actuated_joint_indices: list[int],
    tau_source: str,
    constraint_mode: str,
):
    if tau_source == "actuation_command":
        return actuation_command_tensor(robot)
    if tau_source == "motor_torque":
        return motor_torque_tensor(robot)
    if tau_source == "controller_plus_ground":
        tau = shared_joint_to_full_generalized(robot, motor_torque_tensor(robot))
        tau += shared_projected_contact_sensor_torque(robot, contact_sensor, contact_body_names)
        tau += base_constraint_contact_tensor(robot, constraint_mode)
        return tau
    if tau_source == "applied_torque":
        return robot.data.applied_torque
    if tau_source == "computed_torque":
        return robot.data.computed_torque
    if tau_source == "zero":
        return robot.data.joint_pos * 0.0
    raise ValueError(f"Unsupported record tau source: {tau_source}")


def recording_dynamics_terms(
    robot,
    contact_sensor: ContactSensor,
    contact_body_names: tuple[str, ...],
    tendon_manager: TendonManager,
    joint_acc_for_inertia=None,
    root_acc_for_inertia=None,
    ddq_source: str = "physx_raw",
    include_debug: bool = False,
    constraint_mode: str = "rl_play",
):
    """Compute full-joint inverse-dynamics terms exposed by PhysX for recording."""

    terms = shared_recording_dynamics_terms(
        robot,
        contact_sensor=contact_sensor,
        contact_body_names=contact_body_names,
        tendon_manager=tendon_manager,
        joint_acc_for_inertia=joint_acc_for_inertia,
        root_acc_for_inertia=root_acc_for_inertia,
        ddq_source=ddq_source,
        include_debug=include_debug,
    )
    base_contact = base_constraint_contact_tensor(robot, constraint_mode)
    if torch.any(base_contact).item():
        terms["contact_identification"] = terms["contact_identification"] + base_contact
        terms["base_constraint_contact"] = base_contact
    return terms


def estimated_hip_actuation(robot, actuation_command: torch.Tensor) -> torch.Tensor:
    estimated = torch.zeros_like(actuation_command)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if (
            "_acetabulofemoral_roll" in joint_name
            or "_acetabulofemoral_lateral" in joint_name
            or "_pseudo_acetabulofemoral_flexion" in joint_name
        ):
            estimated[:, joint_index] = actuation_command[:, joint_index]
    return estimated


def estimated_hip_lateral_flexion_actuation(robot, actuation_command: torch.Tensor) -> torch.Tensor:
    estimated = torch.zeros_like(actuation_command)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if "_acetabulofemoral_lateral" in joint_name or "_pseudo_acetabulofemoral_flexion" in joint_name:
            estimated[:, joint_index] = actuation_command[:, joint_index]
    return estimated


def estimated_passive_actuation(robot, actuation_command: torch.Tensor) -> torch.Tensor:
    estimated = torch.zeros_like(actuation_command)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if is_passive_tendon_chain_joint(joint_name):
            estimated[:, joint_index] = actuation_command[:, joint_index]
    return estimated


def passive_solver_constraint(robot, solver_joint: torch.Tensor) -> torch.Tensor:
    constraint = torch.zeros_like(solver_joint)
    for joint_index, joint_name in enumerate(robot.joint_names):
        if is_solver_constraint_diagnostic_joint(joint_name):
            constraint[:, joint_index] = solver_joint[:, joint_index]
    return constraint


def is_passive_tendon_chain_joint(joint_name: str) -> bool:
    return any(
        token in joint_name
        for token in (
            "3b_femorotibial_back",
            "4b_intertarsal_back",
            "3f_femorotibial_front",
            "4f_intertarsal_front",
            "4p_intertarsal_pulley",
            "5_metatarsophalangeal",
            "6_interphalangeal",
        )
    )


def is_solver_constraint_diagnostic_joint(joint_name: str) -> bool:
    return any(
        token in joint_name
        for token in (
            "3b_femorotibial_back",
            "4f_intertarsal_front",
            "4b_intertarsal_back",
            "4p_intertarsal_pulley",
            "5_metatarsophalangeal",
            "8_knee_flexor",
        )
    )


def tendon_joint_torque_tensor(robot, tendon_manager: TendonManager):
    tendon = getattr(tendon_manager, "cached_tendon_joint_torques", None)
    if tendon is None:
        return torch.zeros_like(robot.data.joint_pos)
    return tendon


def pantograph_spring_torque(robot):
    spring = torch.zeros_like(robot.data.joint_pos)
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return spring

    stiffness = robot.data.joint_stiffness[:, pantograph_indices]
    target = robot.data.joint_pos_target[:, pantograph_indices]
    position = robot.data.joint_pos[:, pantograph_indices]
    spring[:, pantograph_indices] = stiffness * (position - target)
    return spring


def pantograph_damping_torque(robot):
    damping_torque = torch.zeros_like(robot.data.joint_pos)
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return damping_torque

    damping = robot.data.joint_damping[:, pantograph_indices]
    target_velocity = robot.data.joint_vel_target[:, pantograph_indices]
    velocity = robot.data.joint_vel[:, pantograph_indices]
    damping_torque[:, pantograph_indices] = damping * (target_velocity - velocity)
    return damping_torque


def pantograph_actuation_torque(robot):
    actuation = torch.zeros_like(robot.data.joint_pos)
    pantograph_indices = [
        index
        for index, joint_name in enumerate(robot.joint_names)
        if joint_name in ("lp1_pantograph", "rp1_pantograph")
    ]
    if not pantograph_indices:
        return actuation

    actuation[:, pantograph_indices] = robot.data.applied_torque[:, pantograph_indices]
    return actuation


def projected_tendon_wrench_torque(robot, tendon_manager: TendonManager):
    link_torques = getattr(tendon_manager, "cached_tendon_link_torques", None)
    link_forces = getattr(tendon_manager, "cached_tendon_forces", None)
    body_ids = getattr(tendon_manager, "cached_tendon_body_ids", None)
    if link_torques is None or body_ids is None:
        return tendon_joint_torque_tensor(robot, tendon_manager)

    body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=robot.device).flatten()
    if body_ids.numel() == 0:
        return torch.zeros_like(robot.data.joint_pos)

    if link_forces is None:
        link_forces = torch.zeros_like(link_torques)

    num_joints = robot.data.joint_pos.shape[1]
    joint_ids = list(range(num_joints))
    jacobian_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    jacobians = robot.root_physx_view.get_jacobians()
    tau_tendon = torch.zeros_like(robot.data.joint_pos)

    for local_body_index, body_id in enumerate(body_ids.tolist()):
        body_index = int(body_id)
        jacobian_body_index = body_index - 1 if robot.is_fixed_base else body_index
        jacobian_linear = jacobians[:, jacobian_body_index, 0:3, :][:, :, jacobian_joint_ids]
        jacobian_angular = jacobians[:, jacobian_body_index, 3:6, :][:, :, jacobian_joint_ids]
        body_quat = robot.data.body_quat_w[:, body_index, :]
        force_world = quat_apply(body_quat, link_forces[:, local_body_index, :])
        torque_world = quat_apply(body_quat, link_torques[:, local_body_index, :])
        tau_tendon += torch.bmm(jacobian_linear.transpose(1, 2), force_world.unsqueeze(-1)).squeeze(-1)
        tau_tendon += torch.bmm(jacobian_angular.transpose(1, 2), torque_world.unsqueeze(-1)).squeeze(-1)

    # The cached link wrenches are opposite the generalized-force sign used by
    # cached_tendon_joint_torques and the database force-balance convention.
    return -tau_tendon


def _actual_generalized_force(robot, joint_ids: list[int], *, force_api_name: str, compensation_api_name: str):
    try:
        force = getattr(robot.root_physx_view, force_api_name)()
        generalized_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
        if force.shape[-1] > max(generalized_joint_ids, default=-1):
            return force[:, generalized_joint_ids]
        return force[:, joint_ids]
    except Exception:
        pass

    compensation = getattr(robot.root_physx_view, compensation_api_name)()
    generalized_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    return -compensation[:, generalized_joint_ids]


def record_side_policy(record_side: str) -> str:
    if record_side == "left":
        return "left_only"
    if record_side == "right":
        return "right_only"
    if record_side == "both":
        return "both_as_samples"
    if record_side == "full":
        return "full_robot"
    raise ValueError(f"Unsupported record side: {record_side}")


def main():  # noqa: C901
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_envs = max(1, int(args_cli.num_envs if args_cli.num_envs is not None else 1))
    env_spacing = float(args_cli.env_spacing if args_cli.env_spacing is not None else 2.5)

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
    print_startup_summary(args_cli, sim_cfg, num_steps, num_envs=num_envs, env_spacing=env_spacing)

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(CAMERA_EYE, CAMERA_TARGET)

    scene_cfg = make_forrest_recording_scene_cfg(num_envs=num_envs, env_spacing=env_spacing)
    scene = InteractiveScene(scene_cfg)
    robot = scene["robot"]
    contact_sensor = scene["contact_forces"]
    contact_body_names = FORREST_PARAMS.training.contacts.contact_sensor_body_names
    for env_id in range(num_envs):
        apply_foot_material(f"/World/envs/env_{env_id}/forrest_isaac")

    debug_left_path = None
    debug_right_path = None
    if not args_cli.jit:
        debug_left_path, debug_right_path = reset_debug_logs(output_dir)
        print(f"Debug logs reset: {debug_left_path}, {debug_right_path}")

    camera, video_writer = setup_video_writer(args_cli, sim_cfg)

    configure_scene_base_constraints(sim, FORREST_PARAMS, args_cli.constraint_mode, num_envs)
    sim.reset()
    reset_robot_to_default(robot, env_origins=scene.env_origins)
    scene.reset()
    scene.update(0.0)

    joint_indices_right, _ = robot.find_joints(joint_names_right, preserve_order=True)
    joint_indices_left, _ = robot.find_joints(joint_names_left, preserve_order=True)

    sim.step()
    scene.update(sim.get_physics_dt())
    time.sleep(0.1)
    previous_record_joint_vel = robot.data.joint_vel.clone()
    previous_record_root_vel = robot.data.root_com_vel_w.clone()
    previous_record_time = 0.0

    actuated_dof_specs = make_actuated_dof_specs(scene_cfg.robot)
    actuated_joint_indices = find_actuated_joint_indices(robot, actuated_dof_specs)
    FORREST_PARAMS.run.controller = args_cli.controller
    left_controller, right_controller = make_leg_controllers(FORREST_PARAMS.run)
    initial_joint_positions = robot.data.joint_pos[:, actuated_joint_indices].clone()
    env_time_offsets = make_env_time_offsets(
        controller=args_cli.controller, num_envs=robot.num_instances, device=robot.device
    )
    startup_hold_duration = max(0.0, float(args_cli.startup_hold_duration)) if args_cli.startup_hold else 0.0

    tendon_data = TendonData(
        robot.num_instances,
        FORREST_PARAMS.to_tendon_randomization_ranges(),
        tc=FORREST_PARAMS.to_tendon_constants(device=robot.device),
        device=robot.device,
    )
    tendon_manager = TendonManager(
        robot,
        tendon_data=tendon_data,
        tendon_damping=FORREST_PARAMS.tendon_damping(),
    )

    calibration_state = None
    calibration_windows = None
    tendon_overlay = None
    if args_cli.calibration:
        calibration_state = build_calibration_state(FORREST_PARAMS, args_cli.controller)
        command_labels = [f"{spec.side}_{spec.dof}" for spec in actuated_dof_specs]
        calibration_windows = CalibrationWindows(
            calibration_state,
            command_labels=command_labels,
            tendon_labels=["left |tau|max", "right |tau|max"],
        )
        tendon_overlay = ForrestTendonOverlay(robot)

    mode_label = "jit" if args_cli.jit else "debug"
    data_recorder = None
    if args_cli.record_identix:
        data_recorder = DataRecording(
            DataRecordingConfig(
                output_dir=args_cli.record_output_dir,
                joint_set=args_cli.record_joint_set,
                side_policy=record_side_policy(args_cli.record_side),
                body_set=args_cli.record_body_set,
                record_base_state=bool(FORREST_PARAMS.recording.record_base_state or args_cli.record_side == "full"),
                record_spatial_state=args_cli.record_spatial_state,
                sampling_stride=args_cli.record_stride,
                startup_skip_seconds=args_cli.record_start_time,
                constraint_mode=args_cli.constraint_mode,
                controller=args_cli.controller,
                tau_source=args_cli.record_tau_source,
                ddq_source=FORREST_PARAMS.recording.ddq_source,
                record_tendons=bool(args_cli.record_tendons and not args_cli.jit),
                record_dynamics=bool(args_cli.record_dynamics),
                record_debug_dynamics=bool(FORREST_PARAMS.recording.record_debug_dynamics),
                record_stabilization_contact=record_stabilization_contact,
                residual_filter_threshold=FORREST_PARAMS.recording.residual_filter_threshold,
                sqlite_filename=FORREST_PARAMS.recording.kinematics_db_filename,
                tendon_sqlite_filename=FORREST_PARAMS.recording.tendons_db_filename,
                dynamics_sqlite_filename=FORREST_PARAMS.recording.dynamics_db_filename,
                debug_sqlite_filename=FORREST_PARAMS.recording.debug_dynamics_db_filename,
                metadata_filename=FORREST_PARAMS.recording.metadata_filename,
                viz_vars_filename=FORREST_PARAMS.recording.viz_vars_filename,
                overwrite=args_cli.record_overwrite,
                parameter_file=args_cli.parameters_file,
            )
        )
        data_recorder.initialize(
            robot,
            sim_dt=sim.get_physics_dt(),
            metadata={
                "mode": mode_label,
                "duration": float(args_cli.duration),
                "num_steps": int(num_steps),
                "num_envs": int(num_envs),
                "env_spacing": float(env_spacing),
                "startup_hold_enabled": bool(args_cli.startup_hold),
                "startup_hold_duration": float(startup_hold_duration),
                "device": args_cli.device,
                "gravity": list(FORREST_PARAMS.physics.gravity),
                "env_time_offsets_s": [float(value) for value in env_time_offsets.detach().cpu().tolist()],
                "base_constraint_contact_policy": (
                    "root body incoming joint wrench folded into contact_identification and controller_plus_ground "
                    "tau for constrained full-base recordings"
                    if record_stabilization_contact
                    else "root body incoming joint wrench not folded into exported contact channel"
                ),
            },
        )
        print(f"Identix recorder initialized: {data_recorder.sqlite_path}")
        if data_recorder.cfg.record_tendons:
            print(f"Tendon visualization recorder initialized: {data_recorder.tendon_sqlite_path}")
        if data_recorder.cfg.record_dynamics:
            print(f"Dynamics recorder initialized: {data_recorder.dynamics_sqlite_path}")

    wall_start = time.perf_counter()
    try:
        iteration = 0
        while iteration < num_steps:
            if calibration_windows is not None:
                calibration_windows.update()

            if calibration_state is not None and calibration_state.consume_reset_request():
                reset_robot_to_default(robot, env_origins=scene.env_origins)
                scene.reset()
                scene.update(0.0)
                tendon_manager.reset_damping_state()
                initial_joint_positions = robot.data.joint_pos[:, actuated_joint_indices].clone()
                previous_record_joint_vel = robot.data.joint_vel.clone()
                previous_record_root_vel = robot.data.root_com_vel_w.clone()
                previous_record_time = 0.0
                iteration = 0
                wall_start = time.perf_counter()
                if calibration_state.should_stop():
                    break

            if calibration_state is not None and calibration_state.is_paused():
                simulation_app.update()
                time.sleep(0.01)
                continue

            t = iteration * sim.get_physics_dt()
            debug_info = None
            debug_joint_pos_left = None
            debug_joint_pos_right = None
            jit_tendon_torques = None

            if calibration_state is not None:
                if calibration_state.consume_tendon_rebuild_request():
                    tendon_data = build_tendon_data_from_state(
                        FORREST_PARAMS,
                        calibration_state,
                        num_instances=robot.num_instances,
                        device=robot.device,
                    )
                    tendon_manager.set_tendon_data(tendon_data)
                apply_tendon_parameters(tendon_data, calibration_state)

            if args_cli.jit:
                jit_tendon_torques = tendon_manager.apply_jit(dt=SIM_DT)
            else:
                # apply_debug reads the current pre-step joint state; keep the JSONL joint_pos aligned with it.
                debug_joint_pos_left = robot.data.joint_pos[0, joint_indices_left].detach().clone()
                debug_joint_pos_right = robot.data.joint_pos[0, joint_indices_right].detach().clone()
                debug_info = tendon_manager.apply_debug(dt=SIM_DT)
                data_left, data_right = leg_tensordict_to_python_dict(debug_info)

            controller_t = max(0.0, t - startup_hold_duration)
            if t < startup_hold_duration:
                commanded_positions = initial_joint_positions
                controller_delta = torch.zeros_like(initial_joint_positions)
            elif calibration_state is not None:
                commanded_positions, controller_delta = batched_runtime_controller_command_tensor(
                    t=controller_t,
                    env_time_offsets=env_time_offsets,
                    state=calibration_state,
                    actuated_dof_specs=actuated_dof_specs,
                    initial_joint_positions=initial_joint_positions,
                )
            else:
                commanded_positions = batched_controller_command_tensor(
                    t=controller_t,
                    env_time_offsets=env_time_offsets,
                    left_controller=left_controller,
                    right_controller=right_controller,
                    actuated_dof_specs=actuated_dof_specs,
                    initial_joint_positions=initial_joint_positions,
                    device=robot.device,
                )
                controller_delta = commanded_positions - initial_joint_positions

            robot.set_joint_position_target(
                commanded_positions,
                joint_ids=actuated_joint_indices,
            )

            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            contact_sensor.update(sim.get_physics_dt(), force_recompute=True)
            recorded_time = (iteration + 1) * sim.get_physics_dt()

            if data_recorder is not None:
                record_dt = max(float(recorded_time) - float(previous_record_time), 1.0e-9)
                joint_acc_recording = (robot.data.joint_vel - previous_record_joint_vel) / record_dt
                root_acc_recording = (robot.data.root_com_vel_w - previous_record_root_vel) / record_dt
                previous_record_joint_vel = robot.data.joint_vel.clone()
                previous_record_root_vel = robot.data.root_com_vel_w.clone()
                previous_record_time = float(recorded_time)
                tau_override = recording_tau_tensor(
                    robot,
                    contact_sensor,
                    contact_body_names,
                    actuated_joint_indices,
                    args_cli.record_tau_source,
                    args_cli.constraint_mode,
                )
                dynamics_terms = None
                if (
                    data_recorder.cfg.record_dynamics
                    or data_recorder.cfg.record_debug_dynamics
                    or data_recorder.cfg.residual_filter_threshold is not None
                ):
                    dynamics_terms = recording_dynamics_terms(
                        robot,
                        contact_sensor,
                        contact_body_names,
                        tendon_manager,
                        joint_acc_for_inertia=joint_acc_recording,
                        root_acc_for_inertia=root_acc_recording,
                        ddq_source=data_recorder.cfg.ddq_source,
                        include_debug=(
                            data_recorder.cfg.record_debug_dynamics
                            or data_recorder.cfg.residual_filter_threshold is not None
                        ),
                        constraint_mode=args_cli.constraint_mode,
                    )
                data_recorder.record_step(
                    step_index=iteration + 1,
                    sim_time=recorded_time,
                    robot=robot,
                    extra_context={"controller_time": controller_t},
                    tau_override=tau_override,
                    ddq_override={"joint_acc": joint_acc_recording, "root_acc": root_acc_recording},
                    dynamics_terms=dynamics_terms,
                )
                if data_recorder.cfg.record_dynamics:
                    if dynamics_terms is None:
                        raise RuntimeError("Internal error: dynamics terms were not computed for dynamics recording.")
                    data_recorder.record_dynamics_step(
                        step_index=iteration + 1,
                        sim_time=recorded_time,
                        robot=robot,
                        dynamics_terms=dynamics_terms,
                        tau_input=tau_override,
                    )

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
                if data_recorder is not None and data_recorder.cfg.record_tendons:
                    data_recorder.record_tendon_frame(
                        step_index=iteration + 1,
                        sim_time=t,
                        side="left",
                        frame=data_left,
                    )
                    data_recorder.record_tendon_frame(
                        step_index=iteration + 1,
                        sim_time=t,
                        side="right",
                        frame=data_right,
                    )
                append_jsonl(debug_left_path, data_left)
                append_jsonl(debug_right_path, data_right)

            if calibration_state is not None:
                tendon_active = {}
                overlay_left_data = data_left if debug_info is not None else None
                overlay_right_data = data_right if debug_info is not None else None
                if debug_info is not None:
                    tendon_plot_values = [
                        debug_info["tendon_torques_left"].detach().abs().max().cpu().item(),
                        debug_info["tendon_torques_right"].detach().abs().max().cpu().item(),
                    ]
                    tendon_active = {
                        "gst": bool(debug_info["GST_not_slack"].detach().any().cpu().item()),
                        "dft": bool(debug_info["DFT_not_slack"].detach().any().cpu().item()),
                        "kft": bool(debug_info["KFT_not_slack"].detach().any().cpu().item()),
                        "edt1": bool(debug_info["EDT1_not_slack"].detach().any().cpu().item()),
                        "edt2": bool(debug_info["EDT2_not_slack"].detach().any().cpu().item()),
                    }
                elif jit_tendon_torques is not None:
                    tendon_plot_values = [
                        jit_tendon_torques[0].detach().abs().max().cpu().item(),
                        jit_tendon_torques[1].detach().abs().max().cpu().item(),
                    ]
                    tendon_active = tendon_manager.get_tendon_activity()
                else:
                    tendon_plot_values = [0.0, 0.0]
                calibration_state.publish_telemetry(
                    sim_time=t,
                    controller_values=controller_delta[0].detach().cpu().tolist(),
                    tendon_values=tendon_plot_values,
                    extra={"tendon_active": tendon_active},
                )
                if tendon_overlay is not None:
                    tendon_overlay.update(
                        iteration=iteration,
                        left_debug=overlay_left_data,
                        right_debug=overlay_right_data,
                        tendon_data=tendon_data,
                        tendon_active=tendon_active,
                    )

            maybe_print_status(
                iteration=iteration,
                num_steps=num_steps,
                sim_time=t,
                wall_start=wall_start,
                status_interval=args_cli.status_interval,
                mode=mode_label,
                debug_info=debug_info,
            )
            iteration += 1

    except KeyboardInterrupt as exc:
        carb.log_error(f"Simulation interrupted: {exc}")
        raise
    finally:
        if tendon_overlay is not None:
            tendon_overlay.clear()
        if calibration_windows is not None:
            calibration_windows.destroy()
        if data_recorder is not None:
            data_recorder.close()
            print(f"Identix database saved to: {data_recorder.sqlite_path}")
            print(f"Identix metadata saved to: {data_recorder.metadata_path}")
            if data_recorder.cfg.record_tendons:
                print(f"Tendon visualization database saved to: {data_recorder.tendon_sqlite_path}")
                print(f"Visualization variables saved to: {data_recorder.viz_vars_path}")
            if data_recorder.cfg.record_dynamics:
                print(f"Dynamics database saved to: {data_recorder.dynamics_sqlite_path}")
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {args_cli.video_output}")

    print("\nCompleted simulation.")
    if not args_cli.jit:
        print(f"Debug logs written to: {debug_left_path} and {debug_right_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
