# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run particle swarm optimization over Forrest tendon/controller parameters.

Example:
    ./isaaclab.sh -p scripts/pso/run.py --headless --jit
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher
from isaaclab.pso.config import PsoConfig
from isaaclab.tendons.parameter_loader import load_forrest_parameter_config, resolve_forrest_config_path

parser = argparse.ArgumentParser(description="Optimize Forrest tendon/controller parameters with PSO.")
parser.add_argument("--pso_config", type=str, default="configs/pso.yaml", help="Path to PSO YAML config.")
parser.add_argument("--parameters_file", type=str, default=None, help="Base Forrest parameter YAML file/profile.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for this PSO run.")
parser.add_argument("--resume", type=str, default=None, help="Resume from a PSO checkpoint.pt.")
parser.add_argument(
    "--reload",
    type=str,
    default=None,
    help=(
        "Reload PSO particle memory from a run directory or checkpoint .pt. Alias for --resume with directory support."
    ),
)
parser.add_argument("--num_particles", type=int, default=None, help="Override swarm particle count.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel simulation env slots.")
parser.add_argument(
    "--rollouts_per_iteration",
    type=int,
    default=None,
    help="Override total rollout evaluations per PSO iteration.",
)
parser.add_argument("--iterations", type=int, default=None, help="Override number of PSO iterations.")
parser.add_argument("--duration", type=float, default=None, help="Override rollout duration in seconds.")
parser.add_argument("--seed", type=int, default=None, help="Override PSO random seed.")
parser.add_argument("--status_interval", type=int, default=None, help="Override status print interval in iterations.")
parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar.")
parser.add_argument("--jit", action="store_true", help="Kept for symmetry with tendon runner; PSO always uses JIT.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

FORREST_CONFIG_PATH = resolve_forrest_config_path(args_cli.parameters_file)
FORREST_PARAMS = load_forrest_parameter_config(FORREST_CONFIG_PATH)
PSO_CFG = PsoConfig.from_yaml(args_cli.pso_config)

if args_cli.num_particles is not None:
    PSO_CFG.swarm.num_particles = args_cli.num_particles
if args_cli.num_envs is not None:
    PSO_CFG.objective.num_envs = args_cli.num_envs
if args_cli.rollouts_per_iteration is not None:
    PSO_CFG.swarm.rollouts_per_iteration = args_cli.rollouts_per_iteration
if args_cli.iterations is not None:
    PSO_CFG.swarm.iterations = args_cli.iterations
if args_cli.duration is not None:
    PSO_CFG.objective.duration = args_cli.duration
if args_cli.seed is not None:
    PSO_CFG.swarm.seed = args_cli.seed
if args_cli.status_interval is not None:
    PSO_CFG.objective.status_interval = args_cli.status_interval
if args_cli.output_dir is not None:
    PSO_CFG.output.directory = args_cli.output_dir

# The Forrest USD intentionally has virtual visual references that USD reports
# once per cloned env. Keep errors visible, but silence this noisy warning class.
if "--/log/channels/omni.usd=" not in args_cli.kit_args:
    args_cli.kit_args = (args_cli.kit_args + " --/log/channels/omni.usd=error").strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.pso.evaluator import ForrestPsoEvaluator
from isaaclab.pso.logging import append_jsonl, tensor_to_list, write_yaml
from isaaclab.pso.optimizer import TorchPso
from isaaclab.pso.parameters import ParameterSpace

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def make_run_dir(base_dir: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_reload_checkpoint(path: str) -> Path:
    reload_path = Path(path).expanduser().resolve()
    if reload_path.is_dir():
        checkpoint_path = reload_path / "checkpoint.pt"
        if checkpoint_path.exists():
            return checkpoint_path
        final_checkpoint_path = reload_path / "checkpoint_final.pt"
        if final_checkpoint_path.exists():
            return final_checkpoint_path
        raise FileNotFoundError(f"Reload directory contains no checkpoint.pt or checkpoint_final.pt: {reload_path}")
    if not reload_path.exists():
        raise FileNotFoundError(f"Reload checkpoint does not exist: {reload_path}")
    return reload_path


def snapshot_base_config(run_dir: Path, base_config_path: Path | None) -> Path | None:
    """Copy the base Forrest config beside PSO outputs so best.yaml is replayable."""

    if base_config_path is None:
        return None
    source = base_config_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Base Forrest parameter config does not exist: {source}")

    if source.is_dir():
        destination = run_dir / "base_config_snapshot"
        if not destination.exists():
            shutil.copytree(source, destination)
        return destination

    suffix = source.suffix or ".yaml"
    destination = run_dir / f"base_config_snapshot{suffix}"
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


def export_best(
    run_dir: Path,
    space: ParameterSpace,
    optimizer: TorchPso,
    *,
    iteration: int,
    base_config_path: Path | None,
    sim_dt: float | None,
    duration: float | None,
    startup_hold_duration: float | None,
    constraint_mode: str | None,
    yaml_path: Path | None = None,
    info_path: Path | None = None,
) -> None:
    best_physical = space.denormalize(optimizer.global_best_position)
    metadata = {
        "best_score": float(optimizer.global_best_score.detach().cpu()),
        "iteration": int(iteration),
        "parameter_names": list(space.names),
        "parameters": space.vector_to_dict(best_physical),
        "sim_dt": None if sim_dt is None else float(sim_dt),
        "duration": None if duration is None else float(duration),
    }
    includes = None if base_config_path is None else [str(base_config_path.resolve())]
    yaml_path = yaml_path or run_dir / "best.yaml"
    info_path = info_path or run_dir / "best_info.yaml"
    exported = space.export_forrest_yaml(yaml_path, best_physical, includes=includes)
    if sim_dt is not None:
        exported.setdefault("physics", {})["sim_dt"] = float(sim_dt)
    run_overrides = exported.setdefault("run", {})
    if duration is not None:
        run_overrides["duration"] = float(duration)
    if startup_hold_duration is not None:
        run_overrides["startup_hold_enabled"] = float(startup_hold_duration) > 0.0
        run_overrides["startup_hold_duration"] = float(startup_hold_duration)
    if constraint_mode is not None:
        run_overrides["constraint_mode"] = str(constraint_mode)
    if sim_dt is not None or duration is not None or startup_hold_duration is not None or constraint_mode is not None:
        write_yaml(yaml_path, exported)
    write_yaml(info_path, metadata)


def export_periodic_best_checkpoint(
    run_dir: Path,
    space: ParameterSpace,
    optimizer: TorchPso,
    *,
    completed_iteration: int,
    base_config_path: Path | None,
    sim_dt: float | None,
    duration: float | None,
) -> None:
    checkpoint_dir = run_dir / "best_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"best_iter_{completed_iteration:06d}"
    export_best(
        run_dir,
        space,
        optimizer,
        iteration=completed_iteration - 1,
        base_config_path=base_config_path,
        sim_dt=sim_dt,
        duration=duration,
        startup_hold_duration=PSO_CFG.objective.startup_hold_duration,
        constraint_mode=PSO_CFG.objective.constraint_mode,
        yaml_path=checkpoint_dir / f"{stem}.yaml",
        info_path=checkpoint_dir / f"{stem}_info.yaml",
    )


def format_iteration_row(
    *,
    iteration: int,
    total_iterations: int,
    iteration_duration_s: float,
    iteration_best_score: float,
    global_best_score: float,
    best_particle_speed: float,
    mean_rollout_speed: float,
    max_rollout_speed: float,
    raw_mean_rollout_speed: float,
    raw_max_rollout_speed: float,
    mean_survival_time: float,
    bad_percent: float,
    fall_percent: float,
    unphysical_percent: float,
    backward_percent: float,
) -> str:
    ok_percent = max(0.0, 100.0 - bad_percent)
    return (
        f"{iteration + 1:>4}/{total_iterations:<4} "
        f"{iteration_duration_s:>8.2f} "
        f"{iteration_best_score:>9.3f} "
        f"{global_best_score:>9.3f} "
        f"{best_particle_speed:>8.2f} "
        f"{mean_rollout_speed:>8.2f} "
        f"{max_rollout_speed:>8.2f} "
        f"{raw_mean_rollout_speed:>8.2f} "
        f"{raw_max_rollout_speed:>8.2f} "
        f"{mean_survival_time:>7.2f} "
        f"{ok_percent:>6.1f} "
        f"{bad_percent:>6.1f} "
        f"{fall_percent:>6.1f} "
        f"{unphysical_percent:>7.1f} "
        f"{backward_percent:>6.1f}"
    )


def format_iteration_header() -> str:
    return (
        "iter    iter_s      score    global   v_best   v_mean    v_max  raw_avg  raw_max  life_s"
        "    ok%   bad%  fall% unphys%  back%\n"
        "------------------------------------------------------------------------------------------------------------------------------"
    )


def main():  # noqa: C901
    if args_cli.resume and args_cli.reload:
        raise ValueError("Use only one of --resume or --reload.")

    reload_checkpoint = None
    if args_cli.resume:
        reload_checkpoint = resolve_reload_checkpoint(args_cli.resume)
    elif args_cli.reload:
        reload_checkpoint = resolve_reload_checkpoint(args_cli.reload)

    if reload_checkpoint is not None:
        run_dir = reload_checkpoint.parent
    else:
        run_dir = make_run_dir(PSO_CFG.output.directory)

    history_path = run_dir / "history.jsonl"
    checkpoint_path = run_dir / "checkpoint.pt"
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    base_config_snapshot_path = snapshot_base_config(run_dir, FORREST_CONFIG_PATH)

    torch.manual_seed(int(PSO_CFG.swarm.seed))
    space = ParameterSpace(PSO_CFG.parameters, device=args_cli.device)
    optimizer = TorchPso(
        PSO_CFG.swarm,
        dim=space.dim,
        device=args_cli.device,
        initial_position=space.initial_normalized(),
    )
    if reload_checkpoint is not None:
        optimizer.load(reload_checkpoint)
        if args_cli.reload and args_cli.iterations is None:
            PSO_CFG.swarm.iterations = int(optimizer.iteration) + int(PSO_CFG.swarm.iterations)

    write_yaml(
        run_dir / "pso_config.yaml",
        {
            "pso": asdict(PSO_CFG),
            "base_parameters_file": None if FORREST_CONFIG_PATH is None else str(FORREST_CONFIG_PATH),
            "base_parameters_snapshot": None if base_config_snapshot_path is None else str(base_config_snapshot_path),
            "device": args_cli.device,
            "reload_checkpoint": None if reload_checkpoint is None else str(reload_checkpoint),
        },
    )

    evaluator = ForrestPsoEvaluator(
        forrest_params=FORREST_PARAMS,
        objective_cfg=PSO_CFG.objective,
        parameter_space=space,
        device=args_cli.device,
        num_particles=PSO_CFG.swarm.num_particles,
        rollouts_per_iteration=PSO_CFG.swarm.rollouts_per_iteration,
    )

    print("\n=== Forrest PSO ===")
    print(f"Particles:          {PSO_CFG.swarm.num_particles}")
    print(f"Parallel envs:      {evaluator.num_envs}")
    print(f"Rollouts/iter:      {evaluator.rollouts_per_iteration}")
    print(f"Async update:       {PSO_CFG.swarm.async_update}")
    print(f"Parameters:         {space.dim}")
    print(f"Iterations:         {PSO_CFG.swarm.iterations}")
    print(f"Topology:           {PSO_CFG.swarm.topology}")
    print(f"Initialization:     {PSO_CFG.swarm.initialization}")
    print(f"Inertia:            {PSO_CFG.swarm.inertia_start} -> {PSO_CFG.swarm.inertia_end}")
    print(f"Restart after:      {PSO_CFG.swarm.restart_after_iterations} stagnant iterations")
    if int(PSO_CFG.swarm.best_reevaluate_interval) > 0:
        print(f"Best re-eval:       every {PSO_CFG.swarm.best_reevaluate_interval} iterations")
    else:
        print("Best re-eval:       disabled")
    print(f"Rollout duration:   {PSO_CFG.objective.duration:.3f} s")
    print(f"Physics dt:         {evaluator.sim_dt:.6f} s")
    print(f"Constraint mode:    {PSO_CFG.objective.constraint_mode}")
    print(f"Device:             {args_cli.device}")
    print(f"Output directory:   {run_dir}")
    if reload_checkpoint is not None:
        print(f"Reloaded state:     {reload_checkpoint}")
    print("Best YAML:          best.yaml")
    print("Particle state:     checkpoint.pt")
    print("Final state:        checkpoint_final.pt")
    print("===================\n")
    print(format_iteration_header(), flush=True)

    progress = None
    if not args_cli.no_progress and tqdm is not None:
        progress = tqdm(
            total=int(PSO_CFG.swarm.iterations),
            initial=int(optimizer.iteration),
            desc="PSO",
            unit="iter",
            dynamic_ncols=True,
        )

    try:
        if bool(PSO_CFG.swarm.async_update):
            remaining_reports = max(0, int(PSO_CFG.swarm.iterations) - int(optimizer.iteration))
            total_async_rollouts = remaining_reports * int(PSO_CFG.swarm.rollouts_per_iteration)
            report_start_s = time.perf_counter()
            for iteration, completed_rollouts_total, improved, result in evaluator.evaluate_async(
                optimizer=optimizer,
                total_rollouts=total_async_rollouts,
                report_interval=int(PSO_CFG.swarm.rollouts_per_iteration),
                total_iterations=int(PSO_CFG.swarm.iterations),
            ):
                iteration_duration_s = time.perf_counter() - report_start_s
                if not improved:
                    optimizer.iterations_since_global_improvement += 1
                iteration_best_score, iteration_best_index = torch.max(result.scores, dim=0)
                iteration_best_index_int = int(iteration_best_index.detach().cpu())
                best_particle_speed = float(result.forward_speed[iteration_best_index_int].detach().cpu())
                global_best_physical = space.denormalize(optimizer.global_best_position)
                curriculum = evaluator.curriculum_summary()
                append_jsonl(
                    history_path,
                    {
                        "iteration": iteration,
                        "mode": "async",
                        "completed_rollouts_total": int(completed_rollouts_total),
                        "iteration_best_score": float(iteration_best_score.detach().cpu()),
                        "iteration_best_particle": iteration_best_index_int,
                        "global_best_score": float(optimizer.global_best_score.detach().cpu()),
                        "global_best_parameters": space.vector_to_dict(global_best_physical),
                        "scores": tensor_to_list(result.scores),
                        "forward_speed": tensor_to_list(result.forward_speed),
                        "forward_displacement": tensor_to_list(result.forward_displacement),
                        "lateral_displacement": tensor_to_list(result.lateral_displacement),
                        "final_height": tensor_to_list(result.final_height),
                        "fell": tensor_to_list(result.fell),
                        "unphysical": tensor_to_list(result.unphysical),
                        "backward": tensor_to_list(result.backward),
                        "terminated": tensor_to_list(result.terminated),
                        "completed_rollouts": result.completed_rollouts,
                        "fall_percent": result.fall_percent,
                        "unphysical_percent": result.unphysical_percent,
                        "backward_percent": result.backward_percent,
                        "terminated_percent": result.terminated_percent,
                        "mean_survival_time": result.mean_survival_time,
                        "mean_rollout_forward_speed": result.mean_rollout_forward_speed,
                        "max_rollout_forward_speed": result.max_rollout_forward_speed,
                        "raw_mean_rollout_forward_speed": result.raw_mean_rollout_forward_speed,
                        "raw_max_rollout_forward_speed": result.raw_max_rollout_forward_speed,
                        "best_particle_forward_speed": best_particle_speed,
                        "command_curriculum": curriculum,
                        "inertia": optimizer.inertia_for_iteration(PSO_CFG.swarm.iterations),
                        "async_updates": optimizer.async_updates,
                        "iterations_since_global_improvement": optimizer.iterations_since_global_improvement,
                    },
                )

                if improved or iteration % max(1, int(PSO_CFG.output.save_every)) == 0:
                    export_best(
                        run_dir,
                        space,
                        optimizer,
                        iteration=iteration,
                        base_config_path=base_config_snapshot_path,
                        sim_dt=PSO_CFG.objective.sim_dt,
                        duration=PSO_CFG.objective.duration,
                        startup_hold_duration=PSO_CFG.objective.startup_hold_duration,
                        constraint_mode=PSO_CFG.objective.constraint_mode,
                    )

                if progress is not None:
                    progress.set_postfix(
                        {
                            "s": f"{float(iteration_best_score.detach().cpu()):.3f}",
                            "g": f"{float(optimizer.global_best_score.detach().cpu()):.3f}",
                            "life": f"{result.mean_survival_time:.2f}s",
                            "cmd": "-" if curriculum is None else f"{int(curriculum['unlocked_bin'])}",
                            "w": f"{optimizer.inertia_for_iteration(PSO_CFG.swarm.iterations):.2f}",
                            "fail%": f"{result.terminated_percent:.1f}",
                        }
                    )
                    progress.update(1)

                status_interval = int(PSO_CFG.objective.status_interval)
                should_print_table = status_interval > 0 and (
                    iteration % status_interval == 0 or iteration == PSO_CFG.swarm.iterations - 1
                )
                if should_print_table:
                    if iteration > 0 and iteration % max(1, status_interval * 20) == 0:
                        if progress is not None:
                            progress.write(format_iteration_header())
                        else:
                            print(format_iteration_header(), flush=True)
                    row = format_iteration_row(
                        iteration=iteration,
                        total_iterations=PSO_CFG.swarm.iterations,
                        iteration_duration_s=iteration_duration_s,
                        iteration_best_score=float(iteration_best_score.detach().cpu()),
                        global_best_score=float(optimizer.global_best_score.detach().cpu()),
                        best_particle_speed=best_particle_speed,
                        mean_rollout_speed=result.mean_rollout_forward_speed,
                        max_rollout_speed=result.max_rollout_forward_speed,
                        raw_mean_rollout_speed=result.raw_mean_rollout_forward_speed,
                        raw_max_rollout_speed=result.raw_max_rollout_forward_speed,
                        mean_survival_time=result.mean_survival_time,
                        bad_percent=result.terminated_percent,
                        fall_percent=result.fall_percent,
                        unphysical_percent=result.unphysical_percent,
                        backward_percent=result.backward_percent,
                    )
                    if progress is not None:
                        progress.write(row)
                    else:
                        print(row, flush=True)

                optimizer.iteration = int(iteration) + 1
                restart_count = optimizer.maybe_restart_stagnant_particles()
                if restart_count > 0:
                    message = f"Restarted {restart_count} stagnant particles at async report {optimizer.iteration}"
                    if progress is not None:
                        progress.write(message)
                    else:
                        print(message, flush=True)
                optimizer.save(checkpoint_path)
                completed_iteration = int(optimizer.iteration)
                best_checkpoint_interval = int(PSO_CFG.output.best_checkpoint_interval)
                if best_checkpoint_interval > 0 and completed_iteration % best_checkpoint_interval == 0:
                    export_periodic_best_checkpoint(
                        run_dir,
                        space,
                        optimizer,
                        completed_iteration=completed_iteration,
                        base_config_path=base_config_snapshot_path,
                        sim_dt=PSO_CFG.objective.sim_dt,
                        duration=PSO_CFG.objective.duration,
                    )
                report_start_s = time.perf_counter()

        while (not bool(PSO_CFG.swarm.async_update)) and optimizer.iteration < int(PSO_CFG.swarm.iterations):
            iteration = optimizer.iteration
            iteration_start_s = time.perf_counter()
            physical = space.denormalize(optimizer.positions)
            result = evaluator.evaluate(physical)
            improved = optimizer.observe(result.scores)
            reevaluated_global_best_score = None

            best_reevaluate_interval = int(PSO_CFG.swarm.best_reevaluate_interval)
            if (
                best_reevaluate_interval > 0
                and (iteration + 1) % best_reevaluate_interval == 0
                and torch.isfinite(optimizer.global_best_score)
            ):
                best_positions = optimizer.global_best_position.unsqueeze(0).expand_as(optimizer.positions)
                reevaluation = evaluator.evaluate(space.denormalize(best_positions))
                finite_scores = reevaluation.scores[torch.isfinite(reevaluation.scores)]
                if finite_scores.numel() > 0:
                    reevaluated_global_best_score = finite_scores.mean()
                    optimizer.blend_global_best_score(reevaluated_global_best_score)

            iteration_best_score, iteration_best_index = torch.max(result.scores, dim=0)
            iteration_best_index_int = int(iteration_best_index.detach().cpu())
            best_particle_speed = float(result.forward_speed[iteration_best_index_int].detach().cpu())
            global_best_physical = space.denormalize(optimizer.global_best_position)
            curriculum = evaluator.curriculum_summary()
            append_jsonl(
                history_path,
                {
                    "iteration": iteration,
                    "iteration_best_score": float(iteration_best_score.detach().cpu()),
                    "iteration_best_particle": iteration_best_index_int,
                    "global_best_score": float(optimizer.global_best_score.detach().cpu()),
                    "global_best_parameters": space.vector_to_dict(global_best_physical),
                    "scores": tensor_to_list(result.scores),
                    "forward_speed": tensor_to_list(result.forward_speed),
                    "forward_displacement": tensor_to_list(result.forward_displacement),
                    "lateral_displacement": tensor_to_list(result.lateral_displacement),
                    "final_height": tensor_to_list(result.final_height),
                    "fell": tensor_to_list(result.fell),
                    "unphysical": tensor_to_list(result.unphysical),
                    "backward": tensor_to_list(result.backward),
                    "terminated": tensor_to_list(result.terminated),
                    "completed_rollouts": result.completed_rollouts,
                    "fall_percent": result.fall_percent,
                    "unphysical_percent": result.unphysical_percent,
                    "backward_percent": result.backward_percent,
                    "terminated_percent": result.terminated_percent,
                    "mean_survival_time": result.mean_survival_time,
                    "mean_rollout_forward_speed": result.mean_rollout_forward_speed,
                    "max_rollout_forward_speed": result.max_rollout_forward_speed,
                    "raw_mean_rollout_forward_speed": result.raw_mean_rollout_forward_speed,
                    "raw_max_rollout_forward_speed": result.raw_max_rollout_forward_speed,
                    "best_particle_forward_speed": best_particle_speed,
                    "command_curriculum": curriculum,
                    "inertia": optimizer.inertia_for_iteration(PSO_CFG.swarm.iterations),
                    "iterations_since_global_improvement": optimizer.iterations_since_global_improvement,
                    "reevaluated_global_best_score": (
                        None
                        if reevaluated_global_best_score is None
                        else float(reevaluated_global_best_score.detach().cpu())
                    ),
                },
            )

            if improved or iteration % max(1, int(PSO_CFG.output.save_every)) == 0:
                export_best(
                    run_dir,
                    space,
                    optimizer,
                    iteration=iteration,
                    base_config_path=base_config_snapshot_path,
                    sim_dt=PSO_CFG.objective.sim_dt,
                    duration=PSO_CFG.objective.duration,
                    startup_hold_duration=PSO_CFG.objective.startup_hold_duration,
                    constraint_mode=PSO_CFG.objective.constraint_mode,
                )

            if progress is not None:
                progress.set_postfix(
                    {
                        "s": f"{float(iteration_best_score.detach().cpu()):.3f}",
                        "g": f"{float(optimizer.global_best_score.detach().cpu()):.3f}",
                        "life": f"{result.mean_survival_time:.2f}s",
                        "cmd": "-" if curriculum is None else f"{int(curriculum['unlocked_bin'])}",
                        "w": f"{optimizer.inertia_for_iteration(PSO_CFG.swarm.iterations):.2f}",
                        "fail%": f"{result.terminated_percent:.1f}",
                    }
                )
                progress.update(1)

            status_interval = int(PSO_CFG.objective.status_interval)
            should_print_table = status_interval > 0 and (
                iteration % status_interval == 0 or iteration == PSO_CFG.swarm.iterations - 1
            )
            if should_print_table:
                if iteration > 0 and iteration % max(1, status_interval * 20) == 0:
                    if progress is not None:
                        progress.write(format_iteration_header())
                    else:
                        print(format_iteration_header(), flush=True)
                row = format_iteration_row(
                    iteration=iteration,
                    total_iterations=PSO_CFG.swarm.iterations,
                    iteration_duration_s=time.perf_counter() - iteration_start_s,
                    iteration_best_score=float(iteration_best_score.detach().cpu()),
                    global_best_score=float(optimizer.global_best_score.detach().cpu()),
                    best_particle_speed=best_particle_speed,
                    mean_rollout_speed=result.mean_rollout_forward_speed,
                    max_rollout_speed=result.max_rollout_forward_speed,
                    raw_mean_rollout_speed=result.raw_mean_rollout_forward_speed,
                    raw_max_rollout_speed=result.raw_max_rollout_forward_speed,
                    mean_survival_time=result.mean_survival_time,
                    bad_percent=result.terminated_percent,
                    fall_percent=result.fall_percent,
                    unphysical_percent=result.unphysical_percent,
                    backward_percent=result.backward_percent,
                )
                if progress is not None:
                    progress.write(row)
                else:
                    print(row, flush=True)

            optimizer.step(total_iterations=PSO_CFG.swarm.iterations)
            restart_count = optimizer.maybe_restart_stagnant_particles()
            if restart_count > 0:
                message = f"Restarted {restart_count} stagnant particles at iter {optimizer.iteration}"
                if progress is not None:
                    progress.write(message)
                else:
                    print(message, flush=True)
            optimizer.save(checkpoint_path)
            completed_iteration = int(optimizer.iteration)
            best_checkpoint_interval = int(PSO_CFG.output.best_checkpoint_interval)
            if best_checkpoint_interval > 0 and completed_iteration % best_checkpoint_interval == 0:
                export_periodic_best_checkpoint(
                    run_dir,
                    space,
                    optimizer,
                    completed_iteration=completed_iteration,
                    base_config_path=base_config_snapshot_path,
                    sim_dt=PSO_CFG.objective.sim_dt,
                    duration=PSO_CFG.objective.duration,
                )

    finally:
        if progress is not None:
            progress.close()
        export_best(
            run_dir,
            space,
            optimizer,
            iteration=max(0, optimizer.iteration - 1),
            base_config_path=base_config_snapshot_path,
            sim_dt=PSO_CFG.objective.sim_dt,
            duration=PSO_CFG.objective.duration,
            startup_hold_duration=PSO_CFG.objective.startup_hold_duration,
            constraint_mode=PSO_CFG.objective.constraint_mode,
        )
        optimizer.save(checkpoint_path)
        optimizer.save(final_checkpoint_path)
        print(f"\nPSO output written to: {run_dir}", flush=True)
        print(
            "Replay best parameters with: ./isaaclab.sh -p scripts/tendons/run.py "
            f"--jit --parameters_file {run_dir / 'best.yaml'}",
            flush=True,
        )
        print(f"Reload particle memory with: ./isaaclab.sh -p scripts/pso/run.py --reload {run_dir}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
