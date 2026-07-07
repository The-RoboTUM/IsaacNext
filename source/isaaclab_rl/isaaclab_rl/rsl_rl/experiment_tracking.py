# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import shutil
import socket
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ExperimentLogger:
    """Small experiment logging interface used by RSL-RL scripts."""

    enabled: bool = False

    def log_scalars(self, scalars: dict[str, float | int], step: int | None = None) -> None:
        raise NotImplementedError

    def log_file(self, path: str | os.PathLike[str], *, name: str | None = None) -> None:
        raise NotImplementedError

    def log_artifact(
        self,
        path: str | os.PathLike[str],
        *,
        name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@dataclass
class TrackingOptions:
    """Frequency controls for optional experiment tracking work."""

    log_interval: int = 1
    extra_metrics_interval: int = 1
    raw_reward_interval: int = 0
    tendon_metrics_interval: int = 10
    checkpoint_alias_interval: int = 1
    checkpoint_artifact_interval: int = 0


class NoOpExperimentLogger(ExperimentLogger):
    """Disabled logger that preserves the same call surface."""

    enabled = False

    def log_scalars(self, scalars: dict[str, float | int], step: int | None = None) -> None:
        return

    def log_file(self, path: str | os.PathLike[str], *, name: str | None = None) -> None:
        return

    def log_artifact(
        self,
        path: str | os.PathLike[str],
        *,
        name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
    ) -> None:
        return

    def close(self) -> None:
        return


class WandbExperimentLogger(ExperimentLogger):
    """W&B-backed logger that can attach to the RSL-RL W&B run lazily."""

    enabled = True

    def __init__(self) -> None:
        import wandb

        self._wandb = wandb
        self._pending_files: list[tuple[str, str | None]] = []
        self._pending_artifacts: list[tuple[str, str, str, list[str] | None]] = []

    def _run(self):
        return getattr(self._wandb, "run", None)

    def _flush_pending(self) -> None:
        if self._run() is None:
            return
        pending_files = self._pending_files
        self._pending_files = []
        for path, name in pending_files:
            self.log_file(path, name=name)
        pending_artifacts = self._pending_artifacts
        self._pending_artifacts = []
        for path, name, artifact_type, aliases in pending_artifacts:
            self.log_artifact(path, name=name, artifact_type=artifact_type, aliases=aliases)

    def log_scalars(self, scalars: dict[str, float | int], step: int | None = None) -> None:
        if not scalars:
            return
        run = self._run()
        if run is None:
            return
        self._wandb.log(scalars, step=step)
        self._flush_pending()

    def log_file(self, path: str | os.PathLike[str], *, name: str | None = None) -> None:
        path_str = str(path)
        if self._run() is None:
            self._pending_files.append((path_str, name))
            return
        kwargs = {"base_path": str(Path(path_str).parent)}
        if name is not None:
            kwargs["policy"] = "now"
        self._wandb.save(path_str, **kwargs)

    def log_artifact(
        self,
        path: str | os.PathLike[str],
        *,
        name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
    ) -> None:
        path_str = str(path)
        if self._run() is None:
            self._pending_artifacts.append((path_str, name, artifact_type, aliases))
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(path_str)
        self._wandb.log_artifact(artifact, aliases=aliases)

    def close(self) -> None:
        self._flush_pending()


def create_experiment_logger(enabled: bool) -> ExperimentLogger:
    """Create the optional experiment logger.

    W&B import errors are intentionally non-fatal so training can continue.
    """

    if not enabled:
        return NoOpExperimentLogger()
    try:
        return WandbExperimentLogger()
    except ModuleNotFoundError:
        print("[WARN] W&B tracking requested but wandb is not installed. Continuing with local files only.")
        return NoOpExperimentLogger()


def wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def write_run_metadata(
    *,
    log_dir: str | os.PathLike[str],
    env_cfg: Any,
    agent_cfg: Any,
    seed: int | None,
    task: str | None,
    tracker: ExperimentLogger,
) -> dict[str, Any]:
    """Collect reproducibility metadata and write it under the run directory."""

    tracking_dir = Path(log_dir) / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    metadata_dict = {
        "run_start_time": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "seed": seed,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "versions": _collect_versions(),
        "gpu": _collect_gpu_info(),
        "git": _collect_git_info(Path.cwd(), tracking_dir),
        "forrest_config": _collect_forrest_config_info(),
        "env_cfg": _to_jsonable(env_cfg),
        "agent_cfg": _to_jsonable(agent_cfg),
    }

    metadata_path = tracking_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_dict, indent=2, sort_keys=True), encoding="utf-8")
    tracker.log_file(metadata_path)
    diff_path = tracking_dir / "git_diff.patch"
    if diff_path.exists():
        tracker.log_file(diff_path)
    return metadata_dict


def collect_checkpoint_infos(runner: Any, env_steps: int = 0) -> dict[str, Any]:
    """Collect extra training state to store in RSL-RL checkpoint infos."""

    infos: dict[str, Any] = {
        "tracking": {
            "env_steps": int(env_steps),
            "rng": _collect_rng_state(),
        }
    }
    curriculum_state = _collect_curriculum_state(getattr(runner, "env", None))
    if curriculum_state is not None:
        infos["tracking"]["curriculum_state"] = curriculum_state
    return infos


def restore_checkpoint_infos(infos: Any) -> int:
    """Restore RNG state from checkpoint infos and return stored environment steps."""

    if not isinstance(infos, dict):
        return 0
    tracking = infos.get("tracking")
    if not isinstance(tracking, dict):
        return 0
    rng_state = tracking.get("rng")
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state)
    return int(tracking.get("env_steps", 0) or 0)


def install_tracking_hooks(
    *,
    runner: Any,
    tracker: ExperimentLogger,
    log_dir: str | os.PathLike[str],
    initial_env_steps: int = 0,
    options: TrackingOptions | None = None,
) -> Any:
    """Install logger and checkpoint hooks on an existing RSL-RL runner."""

    options = options or TrackingOptions()
    needs_extra_tracking = tracker.enabled and any(
        interval > 0
        for interval in (
            options.extra_metrics_interval,
            options.raw_reward_interval,
            options.tendon_metrics_interval,
        )
    )
    needs_logger_proxy = options.log_interval > 1 or needs_extra_tracking
    if needs_logger_proxy:
        tracking_logger = TrackingLoggerProxy(
            wrapped=runner.logger,
            tracker=tracker,
            env=getattr(runner, "env", None),
            log_dir=log_dir,
            initial_env_steps=initial_env_steps,
            options=options,
        )
        runner.logger = tracking_logger
    else:
        tracking_logger = CheckpointTrackingState(
            runner=runner,
            tracker=tracker,
            log_dir=log_dir,
            initial_env_steps=initial_env_steps,
            options=options,
        )

    original_save = runner.save

    def save_with_tracking(path: str, infos: dict | None = None):
        merged_infos = collect_checkpoint_infos(runner, tracking_logger.estimated_env_steps())
        if infos:
            merged_infos.update(infos)
        result = original_save(path, merged_infos)
        if not isinstance(tracking_logger, TrackingLoggerProxy):
            tracking_logger.record_checkpoint(path, getattr(runner, "current_learning_iteration", None))
        return result

    runner.save = save_with_tracking
    runner.tracking_logger = tracking_logger
    return tracking_logger


def finalize_training_checkpoint(runner: Any) -> None:
    """Create final checkpoint aliases and flush the optional tracker."""

    tracking_logger = getattr(runner, "tracking_logger", None)
    if tracking_logger is None:
        return
    tracking_logger.mark_final_checkpoint()
    tracking_logger.close()


class TrackingLoggerProxy:
    """Proxy around RSL-RL's logger that emits stable experiment metric names."""

    def __init__(
        self,
        *,
        wrapped: Any,
        tracker: ExperimentLogger,
        env: Any,
        log_dir: str | os.PathLike[str],
        initial_env_steps: int = 0,
        options: TrackingOptions | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._tracker = tracker
        self._env = env
        self._log_dir = Path(log_dir)
        self._options = options or TrackingOptions()
        self.env_steps = int(initial_env_steps)
        self._reward_sum = None
        self._episode_length = None
        self._returns = deque(maxlen=100)
        self._lengths = deque(maxlen=100)
        self._episode_scalars: dict[str, list[float]] = defaultdict(list)
        self._last_checkpoint: Path | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def init_logging_writer(self) -> None:
        self._wrapped.init_logging_writer()
        self._tracker.close()

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        self._wrapped.process_env_step(rewards, dones, extras, intrinsic_rewards)
        self.env_steps += int(rewards.numel())
        if not self._tracker.enabled or self._options.extra_metrics_interval <= 0:
            return
        self._update_episode_buffers(rewards, dones, intrinsic_rewards)
        self._collect_episode_extras(extras)

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        if _should_run(self._options.log_interval, it):
            self._wrapped.log(
                it,
                start_it,
                total_it,
                collect_time,
                learn_time,
                loss_dict,
                learning_rate,
                action_std,
                rnd_weight,
                print_minimal=print_minimal,
                width=width,
                pad=pad,
            )
        else:
            self._advance_wrapped_logger_without_write(collect_time, learn_time)
        if self._tracker.enabled and self._tracking_scalars_due(it):
            scalars = self._build_training_scalars(it, collect_time, learn_time, loss_dict, learning_rate)
            # RSL-RL's built-in W&B writer uses the learning iteration as the global step.
            # Use the same step here so W&B does not reject later RSL-RL scalar logs as
            # out-of-order. The actual environment-step count is still logged as
            # ``train/env_steps``.
            self._tracker.log_scalars(scalars, step=it)
        self._episode_scalars.clear()

    def save_model(self, path: str, it: int) -> None:
        self._wrapped.save_model(path, it)
        self.record_checkpoint(path, it)

    def stop_logging_writer(self) -> None:
        self._wrapped.stop_logging_writer()
        self._tracker.close()

    def close(self) -> None:
        self._tracker.close()

    def record_checkpoint(self, path: str | os.PathLike[str], it: int | None) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return
        self._last_checkpoint = checkpoint_path
        checkpoint_number = _checkpoint_number(checkpoint_path, it)
        if _should_run(self._options.checkpoint_alias_interval, checkpoint_number):
            latest_path = checkpoint_path.parent / "latest.pt"
            _link_or_copy_file(checkpoint_path, latest_path)
        if self._tracker.enabled and _should_run(self._options.checkpoint_artifact_interval, checkpoint_number):
            self._tracker.log_artifact(
                checkpoint_path,
                name=f"{self._safe_run_name()}-checkpoint",
                artifact_type="checkpoint",
                aliases=[f"iter-{it}", "latest"] if it is not None else ["latest"],
            )

    def mark_final_checkpoint(self) -> None:
        if self._last_checkpoint is None or not self._last_checkpoint.exists():
            candidates = sorted(self._log_dir.glob("model_*.pt"))
            if candidates:
                self._last_checkpoint = candidates[-1]
        if self._last_checkpoint is None:
            return
        final_path = self._last_checkpoint.parent / "final.pt"
        if self._options.checkpoint_alias_interval > 0:
            _link_or_copy_file(self._last_checkpoint, final_path)
        if self._tracker.enabled and self._options.checkpoint_artifact_interval > 0:
            self._tracker.log_artifact(
                final_path,
                name=f"{self._safe_run_name()}-checkpoint",
                artifact_type="checkpoint",
                aliases=["final"],
            )

    def estimated_env_steps(self) -> int:
        return self.env_steps

    def _update_episode_buffers(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        intrinsic_rewards: torch.Tensor | None,
    ) -> None:
        reward_values = rewards.detach()
        if intrinsic_rewards is not None:
            reward_values = reward_values + intrinsic_rewards.detach()
        reward_values = reward_values.reshape(reward_values.shape[0], -1).sum(dim=1)
        done_values = dones.detach().reshape(dones.shape[0], -1).any(dim=1)
        if self._reward_sum is None or self._reward_sum.shape[0] != reward_values.shape[0]:
            self._reward_sum = torch.zeros_like(reward_values)
            self._episode_length = torch.zeros_like(reward_values)
        self._reward_sum += reward_values
        self._episode_length += 1
        done_ids = done_values.nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return
        self._returns.extend(self._reward_sum[done_ids].detach().cpu().tolist())
        self._lengths.extend(self._episode_length[done_ids].detach().cpu().tolist())
        self._reward_sum[done_ids] = 0
        self._episode_length[done_ids] = 0

    def _collect_episode_extras(self, extras: dict) -> None:
        episode = extras.get("episode", extras.get("log", {})) if isinstance(extras, dict) else {}
        if not isinstance(episode, dict):
            return
        for key, value in episode.items():
            scalar = _as_float(value)
            if scalar is not None:
                self._episode_scalars[key].append(scalar)

    def _build_training_scalars(
        self,
        it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
    ) -> dict[str, float | int]:
        collection_size = int(
            self._wrapped.cfg["num_steps_per_env"] * self._wrapped.num_envs * self._wrapped.gpu_world_size
        )
        elapsed = collect_time + learn_time
        scalars: dict[str, float | int] = {
            "train/env_steps": self.env_steps,
            "train/fps": int(collection_size / elapsed) if elapsed > 0 else 0,
            "ppo/learning_rate": float(learning_rate),
        }
        if self._returns:
            scalars["train/return_mean"] = float(np.mean(self._returns))
        if self._lengths:
            scalars["train/episode_length_mean"] = float(np.mean(self._lengths))
        scalars.update(_map_ppo_losses(loss_dict))
        scalars.update(self._map_episode_scalars())
        env = self._unwrap_env()
        if _should_run(self._options.raw_reward_interval, it):
            scalars.update(_extract_current_reward_scalars(env))
        if _should_run(self._options.tendon_metrics_interval, it):
            scalars.update(_extract_tendon_scalars(env))
        return scalars

    def _tracking_scalars_due(self, it: int) -> bool:
        return any(
            (
                _should_run(self._options.extra_metrics_interval, it),
                _should_run(self._options.raw_reward_interval, it),
                _should_run(self._options.tendon_metrics_interval, it),
            )
        )

    def _advance_wrapped_logger_without_write(self, collect_time: float, learn_time: float) -> None:
        collection_size = int(
            self._wrapped.cfg["num_steps_per_env"] * self._wrapped.num_envs * self._wrapped.gpu_world_size
        )
        if getattr(self._wrapped, "writer", None) is not None:
            self._wrapped.tot_timesteps += collection_size
            self._wrapped.tot_time += collect_time + learn_time
        ep_extras = getattr(self._wrapped, "ep_extras", None)
        if ep_extras is not None:
            ep_extras.clear()

    def _map_episode_scalars(self) -> dict[str, float]:
        scalars = {}
        for key, values in self._episode_scalars.items():
            if not values:
                continue
            value = float(np.mean(values))
            if key.startswith(("Episode_Reward/", "Episode_Termination/")):
                continue
            if key.startswith(("Metrics/", "Curriculum/")):
                scalars[key] = value
        return scalars

    def _unwrap_env(self) -> Any:
        env = self._env
        for attr in ("unwrapped", "env"):
            while hasattr(env, attr):
                next_env = getattr(env, attr)
                if next_env is env:
                    break
                env = next_env
        return env

    def _safe_run_name(self) -> str:
        return self._log_dir.name.replace(" ", "_").replace("/", "_")


class CheckpointTrackingState:
    """Save-time-only tracking state used when extra metric logging is disabled."""

    def __init__(
        self,
        *,
        runner: Any,
        tracker: ExperimentLogger,
        log_dir: str | os.PathLike[str],
        initial_env_steps: int,
        options: TrackingOptions,
    ) -> None:
        self._runner = runner
        self._tracker = tracker
        self._log_dir = Path(log_dir)
        self._options = options
        self.env_steps = int(initial_env_steps)
        self._last_checkpoint: Path | None = None

    def estimated_env_steps(self) -> int:
        if self.env_steps > 0:
            return self.env_steps
        iteration = int(getattr(self._runner, "current_learning_iteration", 0) or 0)
        cfg = getattr(self._runner, "cfg", {})
        env = getattr(self._runner, "env", None)
        num_steps = int(cfg.get("num_steps_per_env", 0) or 0)
        num_envs = int(getattr(env, "num_envs", 0) or 0)
        return iteration * num_steps * num_envs

    def record_checkpoint(self, path: str | os.PathLike[str], it: int | None) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return
        self._last_checkpoint = checkpoint_path
        checkpoint_number = _checkpoint_number(checkpoint_path, it)
        if _should_run(self._options.checkpoint_alias_interval, checkpoint_number):
            _link_or_copy_file(checkpoint_path, checkpoint_path.parent / "latest.pt")
        if self._tracker.enabled and _should_run(self._options.checkpoint_artifact_interval, checkpoint_number):
            self._tracker.log_artifact(
                checkpoint_path,
                name=f"{self._safe_run_name()}-checkpoint",
                artifact_type="checkpoint",
                aliases=[f"iter-{it}", "latest"] if it is not None else ["latest"],
            )

    def mark_final_checkpoint(self) -> None:
        if self._last_checkpoint is None or not self._last_checkpoint.exists():
            candidates = sorted(self._log_dir.glob("model_*.pt"))
            if candidates:
                self._last_checkpoint = candidates[-1]
        if self._last_checkpoint is None:
            return
        final_path = self._last_checkpoint.parent / "final.pt"
        if self._options.checkpoint_alias_interval > 0:
            _link_or_copy_file(self._last_checkpoint, final_path)
        if self._tracker.enabled and self._options.checkpoint_artifact_interval > 0:
            self._tracker.log_artifact(
                final_path,
                name=f"{self._safe_run_name()}-checkpoint",
                artifact_type="checkpoint",
                aliases=["final"],
            )

    def close(self) -> None:
        self._tracker.close()

    def _safe_run_name(self) -> str:
        return self._log_dir.name.replace(" ", "_").replace("/", "_")


def _map_ppo_losses(loss_dict: dict) -> dict[str, float]:
    mapping = {
        "surrogate": "ppo/policy_loss",
        "policy": "ppo/policy_loss",
        "value": "ppo/value_loss",
        "entropy": "ppo/entropy",
        "approx_kl": "ppo/approx_kl",
        "clip_fraction": "ppo/clip_fraction",
    }
    scalars = {}
    for key, value in loss_dict.items():
        scalar = _as_float(value)
        if scalar is None:
            continue
        scalars[mapping.get(key, f"ppo/{key}")] = scalar
    return scalars


def _extract_current_reward_scalars(env: Any) -> dict[str, float]:
    reward_manager = getattr(env, "reward_manager", None)
    if reward_manager is None or not hasattr(reward_manager, "_step_reward"):
        return {}
    step_reward = getattr(reward_manager, "_step_reward")
    term_names = getattr(reward_manager, "active_terms", [])
    term_cfgs = getattr(reward_manager, "_term_cfgs", [])
    if step_reward is None or not term_names:
        return {}
    scalars = {}
    for idx, term_name in enumerate(term_names):
        if idx >= step_reward.shape[1]:
            continue
        weighted = step_reward[:, idx]
        scalars[f"reward/weighted_step/{term_name}"] = float(weighted.detach().mean().cpu().item())
        if idx < len(term_cfgs):
            weight = float(getattr(term_cfgs[idx], "weight", 0.0))
            if weight != 0.0:
                scalars[f"reward/raw/{term_name}"] = float((weighted / weight).detach().mean().cpu().item())
    return scalars


def _extract_tendon_scalars(env: Any) -> dict[str, float]:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return {}
    terms = getattr(action_manager, "_terms", None)
    if terms is None:
        terms = getattr(action_manager, "terms", None)
    if isinstance(terms, dict):
        iterable = terms.values()
    elif isinstance(terms, (list, tuple)):
        iterable = terms
    else:
        iterable = []
    for term in iterable:
        tendon_manager = getattr(term, "tendon_manager", None)
        if tendon_manager is not None and hasattr(tendon_manager, "get_tendon_metrics"):
            return tendon_manager.get_tendon_metrics()
    return {}


def _collect_versions() -> dict[str, str | None]:
    packages = {
        "torch": "torch",
        "rsl_rl": "rsl-rl-lib",
        "isaaclab": "isaaclab",
        "isaaclab_rl": "isaaclab_rl",
        "isaacsim": "isaacsim",
        "wandb": "wandb",
    }
    versions = {}
    for key, package_name in packages.items():
        try:
            versions[key] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[key] = None
    versions["cuda"] = torch.version.cuda
    return versions


def _collect_gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    device_index = torch.cuda.current_device()
    return {
        "available": True,
        "device_index": device_index,
        "name": torch.cuda.get_device_name(device_index),
        "device_count": torch.cuda.device_count(),
    }


def _collect_git_info(repo_path: Path, tracking_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "dirty": None, "status": None, "diff_path": None}
    if not (repo_path / ".git").exists():
        return info
    commit = _run_git(repo_path, "rev-parse", "HEAD")
    status = _run_git(repo_path, "status", "--porcelain")
    diff = _run_git(repo_path, "diff", "--binary")
    info["commit"] = commit
    info["dirty"] = bool(status)
    info["status"] = status
    if diff:
        diff_path = tracking_dir / "git_diff.patch"
        diff_path.write_text(diff, encoding="utf-8")
        info["diff_path"] = str(diff_path)
        info["diff_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    return info


def _collect_forrest_config_info() -> dict[str, Any] | None:
    try:
        from isaaclab.tendons.parameter_loader import iter_forrest_config_files, resolve_forrest_config_path
    except Exception:
        return None
    try:
        path = resolve_forrest_config_path()
    except Exception as exc:
        return {"error": str(exc)}
    if path is None:
        return None
    files = []
    try:
        config_files = iter_forrest_config_files(path)
    except Exception:
        config_files = [path]
    for file_path in config_files:
        file_info = {"path": str(file_path)}
        if file_path.exists() and file_path.is_file():
            data = file_path.read_bytes()
            file_info["sha256"] = hashlib.sha256(data).hexdigest()
        files.append(file_info)
    return {"path": str(path), "files": files}


def _collect_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _collect_curriculum_state(env: Any) -> dict[str, Any] | None:
    if env is None:
        return None
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    curriculum_manager = getattr(unwrapped, "curriculum_manager", None)
    if curriculum_manager is None:
        return None
    state = {}
    for attr in ("state_dict", "get_state"):
        fn = getattr(curriculum_manager, attr, None)
        if callable(fn):
            try:
                return _to_jsonable(fn())
            except Exception:
                return None
    for key, value in vars(curriculum_manager).items():
        if key.startswith("_") and key not in {"_term_names"}:
            continue
        state[key] = _to_jsonable(value)
    return state or None


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(np.mean(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _should_run(interval: int, value: int | None) -> bool:
    return interval > 0 and value is not None and value % interval == 0


def _checkpoint_number(path: Path, it: int | None) -> int:
    if it is not None:
        return int(it)
    stem = path.stem
    if stem.startswith("model_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return 0
    return 0


def _link_or_copy_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    try:
        destination.symlink_to(source.name)
        return
    except OSError:
        pass
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, tmp_path)
    tmp_path.replace(destination)


def _run_git(repo_path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
