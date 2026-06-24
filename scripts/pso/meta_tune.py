# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tune PSO hyperparameters with Optuna.

This script does not import Isaac modules or create a SimulationApp.  Each
Optuna trial launches ``scripts/pso/run.py`` in a separate process, then reads
``best_info.yaml`` from that run.  Keeping the process boundary makes failures
and Isaac application lifecycle handling simpler.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed in this Python environment. Install it in the sim env, for example:\n"
        "  /home/robotum/miniconda3/envs/sim/bin/python -m pip install optuna\n"
    ) from exc


parser = argparse.ArgumentParser(description="Meta-optimize Forrest PSO hyperparameters with Optuna.")
parser.add_argument("--meta_config", type=str, default="configs/pso_meta.yaml", help="Path to meta-optimization YAML.")
parser.add_argument("--study_name", type=str, default=None, help="Override Optuna study name.")
parser.add_argument("--storage", type=str, default=None, help="Override Optuna storage URL.")
parser.add_argument("--trials", type=int, default=None, help="Override number of Optuna trials.")
parser.add_argument("--timeout", type=float, default=None, help="Override study timeout in seconds.")
parser.add_argument("--output_dir", type=str, default=None, help="Override meta output directory.")
parser.add_argument("--base_pso_config", type=str, default=None, help="Override base PSO config path.")
parser.add_argument("--parameters_file", type=str, default=None, help="Base Forrest parameter YAML/profile for PSO.")
parser.add_argument("--device", type=str, default=None, help="Device passed to scripts/pso/run.py.")
parser.add_argument(
    "--trial_info",
    choices=("none", "summary", "full"),
    default=None,
    help="How much meta-trial information to print around each child PSO run.",
)
parser.add_argument("--dry_run", action="store_true", help="Write one sampled trial config without launching PSO.")
args = parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level: {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, sort_keys=True) + "\n")


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def require_list(config: dict[str, Any], key: str) -> list[Any]:
    value = config[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"search_space.{key} must be a non-empty list")
    return value


def sample_trial_config(trial: optuna.Trial, base_cfg: dict[str, Any], meta_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, meta_cfg.get("fixed_overrides", {}))
    swarm = cfg.setdefault("swarm", {})
    objective = cfg.setdefault("objective", {})
    search = meta_cfg.get("search_space", {})

    if "num_particles" in search:
        num_particles = int(trial.suggest_categorical("swarm.num_particles", require_list(search, "num_particles")))
        swarm["num_particles"] = num_particles
        swarm["rollouts_per_iteration"] = num_particles
        env_multiplier = int(
            trial.suggest_categorical("objective.num_env_multiplier", require_list(search, "num_env_multiplier"))
        )
        objective["num_envs"] = num_particles * env_multiplier

    inertia_start_low, inertia_start_high = require_list(search, "inertia_start")
    inertia_drop_low, inertia_drop_high = require_list(search, "inertia_drop")
    inertia_start = trial.suggest_float("swarm.inertia_start", float(inertia_start_low), float(inertia_start_high))
    inertia_drop = trial.suggest_float("swarm.inertia_drop", float(inertia_drop_low), float(inertia_drop_high))
    inertia_end = max(float(search.get("inertia_end_min", 0.20)), inertia_start - inertia_drop)
    trial.set_user_attr("swarm.inertia_end", inertia_end)

    cognitive_low, cognitive_high = require_list(search, "cognitive")
    social_low, social_high = require_list(search, "social")
    velocity_low, velocity_high = require_list(search, "velocity_clamp")
    restart_low, restart_high = require_list(search, "restart_fraction")
    swarm["inertia_start"] = inertia_start
    swarm["inertia_end"] = inertia_end
    swarm["cognitive"] = trial.suggest_float("swarm.cognitive", float(cognitive_low), float(cognitive_high))
    swarm["social"] = trial.suggest_float("swarm.social", float(social_low), float(social_high))
    swarm["velocity_clamp"] = trial.suggest_float("swarm.velocity_clamp", float(velocity_low), float(velocity_high))
    swarm["topology"] = trial.suggest_categorical("swarm.topology", require_list(search, "topology"))
    swarm["neighborhood_size"] = int(
        trial.suggest_categorical("swarm.neighborhood_size", require_list(search, "neighborhood_size"))
    )
    swarm["initialization"] = trial.suggest_categorical("swarm.initialization", require_list(search, "initialization"))
    swarm["restart_after_iterations"] = int(
        trial.suggest_categorical("swarm.restart_after_iterations", require_list(search, "restart_after_iterations"))
    )
    swarm["restart_fraction"] = trial.suggest_float("swarm.restart_fraction", float(restart_low), float(restart_high))
    if "best_reevaluate_interval" in search:
        swarm["best_reevaluate_interval"] = int(
            trial.suggest_categorical(
                "swarm.best_reevaluate_interval",
                require_list(search, "best_reevaluate_interval"),
            )
        )
    if "best_reevaluate_blend" in search:
        blend_low, blend_high = require_list(search, "best_reevaluate_blend")
        swarm["best_reevaluate_blend"] = trial.suggest_float(
            "swarm.best_reevaluate_blend",
            float(blend_low),
            float(blend_high),
        )
    return cfg


def run_command(command: list[str], *, log_path: Path, stream_output: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env.get("TERM", "dumb") == "dumb":
        env["TERM"] = "xterm"
    env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if stream_output:
                print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Trial command failed with exit code {return_code}. See log: {log_path}")


def latest_pso_run(run_base: Path) -> Path:
    candidates = [path for path in run_base.iterdir() if path.is_dir() and (path / "best_info.yaml").exists()]
    if not candidates:
        raise FileNotFoundError(f"No PSO run with best_info.yaml found under {run_base}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_last_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last_line = ""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                last_line = line
    return json.loads(last_line) if last_line else {}


def _fmt_float(value: Any, digits: int = 3, missing: str = "-") -> str:
    if value is None:
        return missing
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_trial_start(
    trial: optuna.Trial,
    trial_cfg: dict[str, Any],
    trial_config_path: Path,
    log_path: Path,
    mode: str,
) -> None:
    if mode == "none":
        return
    swarm = trial_cfg.get("swarm", {})
    objective = trial_cfg.get("objective", {})
    print(
        "[meta] "
        f"trial={trial.number} "
        f"particles={swarm.get('num_particles')} "
        f"rollouts={swarm.get('rollouts_per_iteration')} "
        f"envs={objective.get('num_envs')} "
        f"iters={swarm.get('iterations')} "
        f"topology={swarm.get('topology')} "
        f"init={swarm.get('initialization')} "
        f"config={trial_config_path} "
        f"log={log_path}",
        flush=True,
    )
    if mode == "full":
        print(
            "[meta] "
            f"inertia={_fmt_float(swarm.get('inertia_start'))}->{_fmt_float(swarm.get('inertia_end'))} "
            f"cognitive={_fmt_float(swarm.get('cognitive'))} "
            f"social={_fmt_float(swarm.get('social'))} "
            f"vel_clamp={_fmt_float(swarm.get('velocity_clamp'))} "
            f"restart_after={swarm.get('restart_after_iterations')} "
            f"restart_fraction={_fmt_float(swarm.get('restart_fraction'))} "
            f"duration={_fmt_float(objective.get('duration'))}",
            flush=True,
        )


def trial_summary(
    *,
    trial: optuna.Trial,
    value: float,
    trial_cfg: dict[str, Any],
    pso_run_dir: Path,
    history: dict[str, Any],
) -> dict[str, Any]:
    swarm = trial_cfg.get("swarm", {})
    objective = trial_cfg.get("objective", {})
    return {
        "trial": int(trial.number),
        "value": float(value),
        "pso_run_dir": str(pso_run_dir),
        "best_yaml": str(pso_run_dir / "best.yaml"),
        "num_particles": swarm.get("num_particles"),
        "rollouts_per_iteration": swarm.get("rollouts_per_iteration"),
        "num_envs": objective.get("num_envs"),
        "iterations": swarm.get("iterations"),
        "topology": swarm.get("topology"),
        "initialization": swarm.get("initialization"),
        "inertia_start": swarm.get("inertia_start"),
        "inertia_end": swarm.get("inertia_end"),
        "cognitive": swarm.get("cognitive"),
        "social": swarm.get("social"),
        "velocity_clamp": swarm.get("velocity_clamp"),
        "restart_after_iterations": swarm.get("restart_after_iterations"),
        "restart_fraction": swarm.get("restart_fraction"),
        "mean_survival_time": history.get("mean_survival_time"),
        "mean_rollout_forward_speed": history.get("mean_rollout_forward_speed"),
        "max_rollout_forward_speed": history.get("max_rollout_forward_speed"),
        "fall_percent": history.get("fall_percent"),
        "unphysical_percent": history.get("unphysical_percent"),
        "terminated_percent": history.get("terminated_percent"),
        "backward_percent": history.get("backward_percent"),
    }


def print_trial_end(summary: dict[str, Any], mode: str) -> None:
    if mode == "none":
        return
    print(
        "[meta] "
        f"trial={summary['trial']} done "
        f"value={_fmt_float(summary['value'])} "
        f"life={_fmt_float(summary.get('mean_survival_time'), 2)}s "
        f"vmean={_fmt_float(summary.get('mean_rollout_forward_speed'), 2)} "
        f"vmax={_fmt_float(summary.get('max_rollout_forward_speed'), 2)} "
        f"fall={_fmt_float(summary.get('fall_percent'), 1)}% "
        f"unphys={_fmt_float(summary.get('unphysical_percent'), 1)}% "
        f"term={_fmt_float(summary.get('terminated_percent'), 1)}% "
        f"best={summary['best_yaml']}",
        flush=True,
    )
    if mode == "full":
        print(
            "[meta] "
            f"params particles={summary.get('num_particles')} "
            f"rollouts={summary.get('rollouts_per_iteration')} "
            f"envs={summary.get('num_envs')} "
            f"topology={summary.get('topology')} "
            f"init={summary.get('initialization')} "
            f"inertia={_fmt_float(summary.get('inertia_start'))}->{_fmt_float(summary.get('inertia_end'))} "
            f"c1={_fmt_float(summary.get('cognitive'))} "
            f"c2={_fmt_float(summary.get('social'))} "
            f"vclamp={_fmt_float(summary.get('velocity_clamp'))}",
            flush=True,
        )


def objective_factory(meta_cfg: dict[str, Any], base_cfg: dict[str, Any], output_dir: Path):
    parameters_file = meta_cfg.get("parameters_file")
    device = str(meta_cfg.get("device", "cuda:0"))
    headless = bool(meta_cfg.get("headless", True))
    trial_info = str(meta_cfg.get("trial_info", "summary")).lower()

    def objective(trial: optuna.Trial) -> float:
        trial_dir = output_dir / f"trial_{trial.number:05d}"
        trial_run_base = trial_dir / "pso_runs"
        trial_cfg = sample_trial_config(trial, base_cfg, meta_cfg)
        trial_config_path = trial_dir / "pso_trial.yaml"
        write_yaml(trial_config_path, trial_cfg)

        command = [
            "./isaaclab.sh",
            "-p",
            "scripts/pso/run.py",
            "--pso_config",
            str(trial_config_path),
            "--output_dir",
            str(trial_run_base),
            "--device",
            device,
            "--no_progress",
        ]
        if headless:
            command.append("--headless")
        if parameters_file:
            command.extend(["--parameters_file", str(parameters_file)])

        trial.set_user_attr("trial_config", str(trial_config_path))
        trial.set_user_attr("trial_dir", str(trial_dir))
        log_path = trial_dir / "pso.log"
        print_trial_start(trial, trial_cfg, trial_config_path, log_path, trial_info)
        run_command(command, log_path=log_path, stream_output=(trial_info == "full"))

        pso_run_dir = latest_pso_run(trial_run_base)
        best_info = load_yaml(pso_run_dir / "best_info.yaml")
        history = read_last_history(pso_run_dir / "history.jsonl")
        value = float(best_info["best_score"])
        trial.set_user_attr("pso_run_dir", str(pso_run_dir))
        trial.set_user_attr("best_yaml", str(pso_run_dir / "best.yaml"))
        for key in (
            "mean_survival_time",
            "mean_rollout_forward_speed",
            "max_rollout_forward_speed",
            "fall_percent",
            "terminated_percent",
            "backward_percent",
        ):
            if key in history:
                trial.set_user_attr(key, history[key])
        summary = trial_summary(
            trial=trial,
            value=value,
            trial_cfg=trial_cfg,
            pso_run_dir=pso_run_dir,
            history=history,
        )
        append_jsonl(output_dir / "trials_summary.jsonl", summary)
        write_yaml(trial_dir / "trial_summary.yaml", summary)
        print_trial_end(summary, trial_info)
        return value

    return objective


def main() -> None:
    meta_cfg = load_yaml(args.meta_config)
    if args.study_name is not None:
        meta_cfg["study_name"] = args.study_name
    if args.storage is not None:
        meta_cfg["storage"] = args.storage
    if args.trials is not None:
        meta_cfg["n_trials"] = args.trials
    if args.timeout is not None:
        meta_cfg["timeout_s"] = args.timeout
    if args.output_dir is not None:
        meta_cfg["output_dir"] = args.output_dir
    if args.base_pso_config is not None:
        meta_cfg["base_pso_config"] = args.base_pso_config
    if args.parameters_file is not None:
        meta_cfg["parameters_file"] = args.parameters_file
    if args.device is not None:
        meta_cfg["device"] = args.device
    if args.trial_info is not None:
        meta_cfg["trial_info"] = args.trial_info

    output_dir = Path(meta_cfg["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "pso_meta_config.yaml", meta_cfg)
    base_cfg = load_yaml(meta_cfg["base_pso_config"])

    if args.dry_run:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        trial_cfg = sample_trial_config(trial, base_cfg, meta_cfg)
        dry_run_path = output_dir / "dry_run_trial.yaml"
        write_yaml(dry_run_path, trial_cfg)
        print(f"Wrote dry-run trial config: {dry_run_path}")
        return

    study = optuna.create_study(
        study_name=str(meta_cfg["study_name"]),
        storage=str(meta_cfg["storage"]),
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        objective_factory(meta_cfg, base_cfg, output_dir),
        n_trials=int(meta_cfg.get("n_trials", 20)),
        timeout=None if meta_cfg.get("timeout_s") is None else float(meta_cfg["timeout_s"]),
    )

    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_user_attrs": dict(study.best_trial.user_attrs),
    }
    write_yaml(output_dir / "best_meta.yaml", best)
    print(f"Best meta result written to: {output_dir / 'best_meta.yaml'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
