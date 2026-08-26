# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a generic dynamics residual audit on simple IsaacLab articulations.

Example:

.. code-block:: bash

    ./isaaclab.sh -p scripts/dynamics_audit/run_articulation_audit.py \
        --asset cartpole --num_envs 4 --num_steps 1000 --headless
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run simple-articulation dynamics force-balance audits.")
parser.add_argument("--asset", choices=("cartpole", "cart_double_pendulum"), default="cartpole")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of recorded simulation steps.")
parser.add_argument("--warmup_steps", type=int, default=8, help="Steps to simulate before recording.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for audit DB and metadata.")
parser.add_argument("--overwrite", action="store_true", help="Replace an existing audit DB in output_dir.")
parser.add_argument(
    "--control_mode",
    choices=("zero", "sinusoid", "random", "impulse"),
    default="sinusoid",
    help="Deterministic effort command pattern.",
)
parser.add_argument("--action_scale", type=float, default=None, help="Override effort amplitude.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--actuate_all_joints", action="store_true", help="Apply commands to every joint.")
parser.add_argument("--no_mass_matrix", action="store_true", help="Do not store mass matrix columns.")
parser.add_argument("--include_base", action="store_true", help="Keep floating-base generalized slots when present.")
parser.add_argument(
    "--asset_drive_gains",
    action="store_true",
    help="Keep the asset's original implicit stiffness/damping instead of zeroing them for a pure effort audit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # isort:skip

import isaaclab.sim as sim_utils  # isort:skip
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # isort:skip
from isaaclab.dynamics_audit import DynamicsAuditRecorder, compute_articulation_dynamics_terms  # isort:skip
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # isort:skip
from isaaclab.sim import SimulationContext  # isort:skip
from isaaclab.utils import configclass  # isort:skip
from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG  # isort:skip
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


@configclass
class AuditSceneCfg(InteractiveSceneCfg):
    """Scene used by the standalone dynamics audit."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = AuditSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene_cfg.robot = _asset_cfg(args_cli.asset).replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Simple dynamics audit scene setup complete.")

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    _reset_scene(scene, robot)
    previous_joint_vel = robot.data.joint_vel.clone()
    previous_command = torch.zeros_like(robot.data.joint_pos)
    output_dir = _output_dir(args_cli.asset)
    if args_cli.overwrite:
        for filename in ("dynamics_audit.db", "metadata.json"):
            path = output_dir / filename
            if path.exists():
                path.unlink()
    recorder = DynamicsAuditRecorder(
        output_dir,
        asset_name=args_cli.asset,
        coordinate_names=_coordinate_names(robot, include_base=bool(args_cli.include_base)),
        num_envs=args_cli.num_envs,
        include_mass_matrix=not args_cli.no_mass_matrix,
        metadata={
            "control_mode": args_cli.control_mode,
            "action_scale": _action_scale(args_cli.asset),
            "warmup_steps": args_cli.warmup_steps,
            "physics_dt": float(sim_dt),
            "seed": int(args_cli.seed),
            "joint_names": list(robot.joint_names),
            "actuated_joint_names": _actuated_joint_names(args_cli.asset, list(robot.joint_names)),
            "include_base": bool(args_cli.include_base),
            "zeroed_implicit_drive_gains": not bool(args_cli.asset_drive_gains),
            "joint_stiffness": robot.data.joint_stiffness.detach().cpu().tolist(),
            "joint_damping": robot.data.joint_damping.detach().cpu().tolist(),
        },
    )

    total_steps = int(args_cli.warmup_steps) + int(args_cli.num_steps)
    for step_index in range(total_steps):
        command = _command_effort(
            args_cli.asset,
            robot,
            step_index=step_index,
            dt=float(sim_dt),
            mode=args_cli.control_mode,
            actuate_all_joints=bool(args_cli.actuate_all_joints),
        )
        robot.set_joint_effort_target(command)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step_index >= int(args_cli.warmup_steps):
            terms = compute_articulation_dynamics_terms(
                robot,
                dt=float(sim_dt),
                command_effort=command,
                previous_joint_vel=previous_joint_vel,
                previous_command_effort=previous_command,
                include_base=bool(args_cli.include_base),
            )
            recorder.record(step_index=step_index, time=step_index * float(sim_dt), terms=terms)
        previous_joint_vel = robot.data.joint_vel.clone()
        previous_command = command.clone()

    recorder.close(print_report=True)


def _asset_cfg(asset_name: str):
    if asset_name == "cartpole":
        cfg = CARTPOLE_CFG.copy()
    elif asset_name == "cart_double_pendulum":
        cfg = CART_DOUBLE_PENDULUM_CFG.copy()
    else:
        raise ValueError(f"Unsupported asset: {asset_name}")
    if not bool(args_cli.asset_drive_gains):
        _zero_implicit_drive_gains(cfg)
    return cfg


def _zero_implicit_drive_gains(cfg) -> None:
    for actuator_cfg in cfg.actuators.values():
        actuator_cfg.stiffness = 0.0
        actuator_cfg.damping = 0.0


def _reset_scene(scene: InteractiveScene, robot) -> None:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    scene.reset()


def _output_dir(asset_name: str) -> Path:
    if args_cli.output_dir:
        return Path(args_cli.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "dynamics_audit" / f"{asset_name}_{stamp}"


def _coordinate_names(robot, *, include_base: bool) -> list[str]:
    if include_base and not bool(getattr(robot, "is_fixed_base", True)):
        return [
            "base_pos_x",
            "base_pos_y",
            "base_pos_z",
            "base_roll",
            "base_pitch",
            "base_yaw",
            *list(robot.joint_names),
        ]
    return list(robot.joint_names)


def _action_scale(asset_name: str) -> float:
    if args_cli.action_scale is not None:
        return float(args_cli.action_scale)
    if asset_name == "cart_double_pendulum":
        return 35.0
    return 12.0


def _actuated_joint_names(asset_name: str, joint_names: list[str]) -> list[str]:
    if bool(args_cli.actuate_all_joints):
        return list(joint_names)
    if asset_name == "cartpole":
        return [name for name in joint_names if name == "slider_to_cart"]
    if asset_name == "cart_double_pendulum":
        return [name for name in joint_names if name in ("slider_to_cart", "pole_to_pendulum")]
    return list(joint_names)


def _command_effort(
    asset_name: str,
    robot,
    *,
    step_index: int,
    dt: float,
    mode: str,
    actuate_all_joints: bool,
) -> torch.Tensor:
    command = torch.zeros_like(robot.data.joint_pos)
    actuated_names = (
        list(robot.joint_names) if actuate_all_joints else _actuated_joint_names(asset_name, list(robot.joint_names))
    )
    actuated_indices = [index for index, name in enumerate(robot.joint_names) if name in set(actuated_names)]
    if not actuated_indices or mode == "zero":
        return command
    scale = _action_scale(asset_name)
    t = float(step_index) * float(dt)
    if mode == "sinusoid":
        for local_index, joint_index in enumerate(actuated_indices):
            phase = 0.7 * float(local_index)
            command[:, joint_index] = scale * torch.sin(torch.full_like(command[:, joint_index], 2.0 * t + phase))
    elif mode == "random":
        command[:, actuated_indices] = scale * (2.0 * torch.rand_like(command[:, actuated_indices]) - 1.0)
    elif mode == "impulse":
        period = 50
        sign = 1.0 if (step_index // period) % 2 == 0 else -1.0
        command[:, actuated_indices] = sign * scale
    else:
        raise ValueError(f"Unsupported control mode: {mode}")
    return command


if __name__ == "__main__":
    main()
    simulation_app.close()
