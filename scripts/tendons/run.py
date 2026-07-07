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
    choices=("left", "right", "both"),
    default=None,
    help="Leg side to record. 'both' stores each side as separate samples.",
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
    choices=("controller_plus_ground", "applied_torque", "computed_torque", "zero"),
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
if args_cli.record_output_dir is None:
    args_cli.record_output_dir = FORREST_PARAMS.recording.output_dir
if args_cli.record_output_dir is None:
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args_cli.record_output_dir = str(Path(args_cli.output_dir) / f"forrest_dbs_{datestamp}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.tendons.data_recording import DataRecording, DataRecordingConfig
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
        print(f"Recording stride:  {args.record_stride}")
        print(f"Recording tau:     {args.record_tau_source}")
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


def projected_contact_sensor_torque(robot, contact_sensor: ContactSensor, contact_body_names: tuple[str, ...]):
    """Project measured world-frame contact forces into generalized joint torques."""

    num_joints = robot.data.joint_pos.shape[1]
    tau_ground = torch.zeros_like(robot.data.joint_pos)
    sensor_body_indices = [contact_sensor.body_names.index(name) for name in contact_body_names]
    robot_body_indices, _ = robot.find_bodies(list(contact_body_names), preserve_order=True)

    if contact_sensor.data.force_matrix_w is not None:
        forces_world = contact_sensor.data.force_matrix_w[:, sensor_body_indices, 0, :]
    else:
        forces_world = contact_sensor.data.net_forces_w[:, sensor_body_indices, :]
    if not torch.any(forces_world).item():
        return tau_ground

    joint_ids = list(range(num_joints))
    jacobian_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]
    jacobians = robot.root_physx_view.get_jacobians()

    for local_foot_index, body_index in enumerate(robot_body_indices):
        jacobian_body_index = int(body_index) - 1 if robot.is_fixed_base else int(body_index)
        jacobian_pos = jacobians[:, jacobian_body_index, 0:3, :][:, :, jacobian_joint_ids]
        tau_ground += torch.bmm(
            jacobian_pos.transpose(1, 2), forces_world[:, local_foot_index, :].unsqueeze(-1)
        ).squeeze(-1)

    return tau_ground


def recording_tau_tensor(
    robot,
    contact_sensor: ContactSensor,
    contact_body_names: tuple[str, ...],
    actuated_joint_indices: list[int],
    tau_source: str,
):
    if tau_source == "controller_plus_ground":
        tau = torch.zeros_like(robot.data.joint_pos)
        tau[:, actuated_joint_indices] = robot.data.applied_torque[:, actuated_joint_indices]
        tau += projected_contact_sensor_torque(robot, contact_sensor, contact_body_names)
        return tau
    if tau_source == "applied_torque":
        return robot.data.applied_torque
    if tau_source == "computed_torque":
        return robot.data.computed_torque
    if tau_source == "zero":
        return robot.data.joint_pos * 0.0
    raise ValueError(f"Unsupported record tau source: {tau_source}")


def recording_dynamics_terms(robot):
    """Compute full-joint inverse-dynamics terms exposed by PhysX for recording."""

    num_joints = robot.data.joint_pos.shape[1]
    joint_ids = list(range(num_joints))
    generalized_joint_ids = joint_ids if robot.is_fixed_base else [joint_id + 6 for joint_id in joint_ids]

    mass_matrices = robot.root_physx_view.get_generalized_mass_matrices()
    if not robot.is_fixed_base:
        mass_matrices = mass_matrices[:, 6:, 6:]
    inertia = torch.bmm(mass_matrices, robot.data.joint_acc.unsqueeze(-1)).squeeze(-1)

    coriolis_all = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
    gravity_all = robot.root_physx_view.get_gravity_compensation_forces()
    coriolis = coriolis_all[:, generalized_joint_ids]
    gravity = gravity_all[:, generalized_joint_ids]

    dynamic = robot.data.joint_dynamic_friction_coeff
    viscous = robot.data.joint_viscous_friction_coeff
    friction = -dynamic * torch.sign(robot.data.joint_vel) - viscous * robot.data.joint_vel

    return {
        "inertia": inertia,
        "coriolis": coriolis,
        "gravity": gravity,
        "friction": friction,
    }


def record_side_policy(record_side: str) -> str:
    if record_side == "left":
        return "left_only"
    if record_side == "right":
        return "right_only"
    if record_side == "both":
        return "both_as_samples"
    raise ValueError(f"Unsupported record side: {record_side}")


def main():  # noqa: C901
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
    contact_body_names = FORREST_PARAMS.training.contacts.foot_body_names
    contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=f"{robot_cfg.prim_path}/{body_name_regex(contact_body_names)}",
            update_period=sim_cfg.dt,
            history_length=1,
            track_air_time=False,
            filter_prim_paths_expr=["/World/defaultGroundPlane/GroundPlane/CollisionPlane"],
        )
    )

    ground_cfg = sim_utils.GroundPlaneCfg(physics_material=make_ground_material_cfg())
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    apply_foot_material("/World/Bot")
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
    initial_joint_positions = robot.data.joint_pos[:, actuated_joint_indices].clone()
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
                record_spatial_state=args_cli.record_spatial_state,
                sampling_stride=args_cli.record_stride,
                startup_skip_seconds=args_cli.record_start_time,
                constraint_mode=args_cli.constraint_mode,
                controller=args_cli.controller,
                tau_source=args_cli.record_tau_source,
                record_tendons=bool(args_cli.record_tendons and not args_cli.jit),
                record_dynamics=bool(args_cli.record_dynamics),
                sqlite_filename=FORREST_PARAMS.recording.kinematics_db_filename,
                tendon_sqlite_filename=FORREST_PARAMS.recording.tendons_db_filename,
                dynamics_sqlite_filename=FORREST_PARAMS.recording.dynamics_db_filename,
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
                "startup_hold_enabled": bool(args_cli.startup_hold),
                "startup_hold_duration": float(startup_hold_duration),
                "device": args_cli.device,
                "gravity": list(FORREST_PARAMS.physics.gravity),
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
                reset_robot_to_default(robot)
                robot.update(0.0)
                tendon_manager.reset_damping_state()
                initial_joint_positions = robot.data.joint_pos[:, actuated_joint_indices].clone()
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
                commanded_positions, controller_delta = runtime_controller_command_tensor(
                    t=controller_t,
                    state=calibration_state,
                    actuated_dof_specs=actuated_dof_specs,
                    initial_joint_positions=initial_joint_positions,
                )
            else:
                commanded_positions = controller_command_tensor(
                    t=controller_t,
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
                tau_override = recording_tau_tensor(
                    robot,
                    contact_sensor,
                    contact_body_names,
                    actuated_joint_indices,
                    args_cli.record_tau_source,
                )
                data_recorder.record_step(
                    step_index=iteration + 1,
                    sim_time=recorded_time,
                    robot=robot,
                    extra_context={"controller_time": controller_t},
                    tau_override=tau_override,
                )
                if data_recorder.cfg.record_dynamics:
                    data_recorder.record_dynamics_step(
                        step_index=iteration + 1,
                        sim_time=recorded_time,
                        robot=robot,
                        dynamics_terms=recording_dynamics_terms(robot),
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
