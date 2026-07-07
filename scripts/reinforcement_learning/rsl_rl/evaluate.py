# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic checkpoint evaluation for RSL-RL agents."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint deterministically.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
parser.add_argument("--episodes", type=int, default=32, help="Number of completed episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum environment steps before stopping.")
parser.add_argument("--output", type=str, default=None, help="Output JSON path. Defaults to the checkpoint run folder.")
parser.add_argument("--lin_vel_x", type=float, default=None, help="Fixed base velocity x command, if supported.")
parser.add_argument("--lin_vel_y", type=float, default=None, help="Fixed base velocity y command, if supported.")
parser.add_argument("--ang_vel_z", type=float, default=None, help="Fixed yaw velocity command, if supported.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    create_experiment_logger,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
    wandb_available,
)
from isaaclab_rl.rsl_rl.experiment_tracking import _extract_tendon_scalars
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL agent checkpoint."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_fixed_velocity_command(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    max_steps = args_cli.max_steps or int(env.unwrapped.max_episode_length * max(1, args_cli.episodes))

    returns = torch.zeros(env.num_envs, device=env.unwrapped.device)
    lengths = torch.zeros(env.num_envs, device=env.unwrapped.device)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    termination_counts: dict[str, float] = {}
    command_errors: list[float] = []
    tendon_snapshots: list[dict[str, float]] = []

    steps = 0
    while simulation_app.is_running() and len(completed_returns) < args_cli.episodes and steps < max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0") and hasattr(policy, "reset"):
                policy.reset(dones)

        rewards_flat = rewards.reshape(rewards.shape[0], -1).sum(dim=1)
        dones_flat = dones.reshape(dones.shape[0], -1).any(dim=1)
        returns += rewards_flat
        lengths += 1
        _accumulate_termination_counts(termination_counts, extras)
        command_error = _command_tracking_error(env.unwrapped)
        if command_error is not None:
            command_errors.append(command_error)
        tendon_metrics = _extract_tendon_scalars(env.unwrapped)
        if tendon_metrics:
            tendon_snapshots.append(tendon_metrics)

        done_ids = dones_flat.nonzero(as_tuple=False).flatten()
        for env_id in done_ids.tolist():
            if len(completed_returns) >= args_cli.episodes:
                break
            completed_returns.append(float(returns[env_id].cpu().item()))
            completed_lengths.append(float(lengths[env_id].cpu().item()))
        if done_ids.numel() > 0:
            returns[done_ids] = 0.0
            lengths[done_ids] = 0.0
        steps += 1

    result = _summarize_results(
        checkpoint=resume_path,
        completed_returns=completed_returns,
        completed_lengths=completed_lengths,
        termination_counts=termination_counts,
        command_errors=command_errors,
        tendon_snapshots=tendon_snapshots,
        steps=steps,
    )
    output_path = Path(args_cli.output) if args_cli.output else Path(log_dir) / "eval" / _default_eval_filename()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] Evaluation results written to: {output_path}")

    tracker = create_experiment_logger(enabled=agent_cfg.logger == "wandb" and wandb_available())
    tracker.log_file(output_path)
    tracker.log_scalars(_flatten_eval_scalars(result), step=steps * env.num_envs)
    tracker.close()
    env.close()


def _apply_fixed_velocity_command(env_cfg) -> None:
    command_values = {
        "lin_vel_x": args_cli.lin_vel_x,
        "lin_vel_y": args_cli.lin_vel_y,
        "ang_vel_z": args_cli.ang_vel_z,
    }
    if all(value is None for value in command_values.values()):
        return
    commands_cfg = getattr(env_cfg, "commands", None)
    if commands_cfg is None or not hasattr(commands_cfg, "base_velocity"):
        print("[WARN] Fixed velocity command requested, but this environment has no base_velocity command config.")
        return
    base_velocity = commands_cfg.base_velocity
    ranges = getattr(base_velocity, "ranges", None)
    if ranges is None:
        print("[WARN] Fixed velocity command requested, but base_velocity has no ranges field.")
        return
    for name, value in command_values.items():
        if value is not None and hasattr(ranges, name):
            setattr(ranges, name, (float(value), float(value)))


def _accumulate_termination_counts(counts: dict[str, float], extras: dict) -> None:
    episode = extras.get("episode", extras.get("log", {})) if isinstance(extras, dict) else {}
    if not isinstance(episode, dict):
        return
    for key, value in episode.items():
        if not key.startswith("Episode_Termination/"):
            continue
        term = key.split("/", 1)[1]
        scalar = _as_float(value)
        if scalar is not None:
            counts[term] = counts.get(term, 0.0) + scalar


def _command_tracking_error(env) -> float | None:
    command_manager = getattr(env, "command_manager", None)
    robot = getattr(getattr(env, "scene", None), "articulations", {}).get("robot") if hasattr(env, "scene") else None
    if command_manager is None or robot is None:
        return None
    try:
        command = command_manager.get_command("base_velocity")
    except Exception:
        return None
    actual = robot.data.root_lin_vel_b[:, :2]
    target = command[:, :2]
    return float(torch.linalg.norm(target - actual, dim=1).mean().cpu().item())


def _summarize_results(
    *,
    checkpoint: str,
    completed_returns: list[float],
    completed_lengths: list[float],
    termination_counts: dict[str, float],
    command_errors: list[float],
    tendon_snapshots: list[dict[str, float]],
    steps: int,
) -> dict:
    result = {
        "checkpoint": checkpoint,
        "episodes": len(completed_returns),
        "steps": steps,
        "return_mean": float(np.mean(completed_returns)) if completed_returns else None,
        "return_std": float(np.std(completed_returns)) if completed_returns else None,
        "episode_length_mean": float(np.mean(completed_lengths)) if completed_lengths else None,
        "episode_length_std": float(np.std(completed_lengths)) if completed_lengths else None,
        "termination_counts": termination_counts,
        "command_tracking_error_mean": float(np.mean(command_errors)) if command_errors else None,
        "tendon": {},
    }
    if tendon_snapshots:
        keys = sorted({key for snapshot in tendon_snapshots for key in snapshot})
        result["tendon"] = {
            key: float(np.mean([snapshot[key] for snapshot in tendon_snapshots if key in snapshot])) for key in keys
        }
    return result


def _flatten_eval_scalars(result: dict) -> dict[str, float]:
    scalars = {}
    keys = (
        "return_mean",
        "return_std",
        "episode_length_mean",
        "episode_length_std",
        "command_tracking_error_mean",
    )
    for key in keys:
        if result.get(key) is not None:
            scalars[f"eval/{key}"] = float(result[key])
    for key, value in result.get("termination_counts", {}).items():
        scalars[f"eval/termination/{key}"] = float(value)
    for key, value in result.get("tendon", {}).items():
        scalars[f"eval/{key}"] = float(value)
    return scalars


def _as_float(value) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _default_eval_filename() -> str:
    return f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"


if __name__ == "__main__":
    main()
    simulation_app.close()
