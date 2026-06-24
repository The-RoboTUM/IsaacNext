# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration loading for Forrest PSO runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from isaaclab.curriculums.command_bins import CommandBinCurriculumParameters


@dataclass
class ParameterSpec:
    """One scalar optimization parameter."""

    name: str
    min: float
    max: float
    initial: float | None = None


@dataclass
class SwarmConfig:
    num_particles: int = 32
    rollouts_per_iteration: int = 64
    iterations: int = 20
    async_update: bool = False
    inertia: float = 0.70
    inertia_start: float | None = 0.90
    inertia_end: float | None = 0.40
    cognitive: float = 1.50
    social: float = 1.50
    velocity_clamp: float = 0.25
    topology: str = "ring"
    neighborhood_size: int = 5
    initialization: str = "sobol"
    restart_after_iterations: int = 10
    restart_fraction: float = 0.25
    reset_personal_best_on_restart: bool = True
    best_reevaluate_interval: int = 10
    best_reevaluate_blend: float = 0.50
    seed: int = 42


@dataclass
class ObjectiveConfig:
    num_envs: int | None = None
    duration: float = 6.0
    sim_dt: float | None = None
    startup_hold_duration: float = 1.5
    constraint_mode: str = "boom"
    env_spacing: float = 3.0
    status_interval: int = 1
    eval_after_startup_hold: bool = True
    replicate_physics: bool = False
    use_command_curriculum: bool = True
    command_curriculum: CommandBinCurriculumParameters = field(default_factory=CommandBinCurriculumParameters)
    terminations: TerminationConfig = field(default_factory=lambda: TerminationConfig())
    prefer_newer_command_bins: bool = True
    older_command_bin_probability_decay: float = 0.35
    command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    reward_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class OutputConfig:
    directory: str = "outputs/pso"
    save_every: int = 1
    best_checkpoint_interval: int = 10


@dataclass
class TerminationConfig:
    base_too_low_height: float = 0.2
    terminate_on_unphysical: bool = True
    unphysical_penalty: float = 12.0
    terminate_on_backward_progress: bool = False
    backward_termination_grace_s: float = 1.0
    max_backward_displacement: float = 0.25
    max_forward_speed: float = 8.0
    max_lateral_speed: float = 1.5
    max_vertical_speed: float = 3.0
    max_root_angular_speed: float = 12.0
    max_height: float = 2.2
    terminate_on_undesired_contact: bool = True
    undesired_contact_grace_s: float = 0.10
    undesired_contact_consecutive_steps: int = 2
    undesired_contact_force_threshold: float = 1.0
    undesired_contact_body_names: tuple[str, ...] = ()
    terminate_on_joint_vibration: bool = True
    joint_vibration_grace_s: float = 0.25
    joint_vibration_consecutive_steps: int = 2
    max_joint_vibration_velocity: float = 60.0
    max_joint_vibration_acceleration: float = 6000.0
    joint_vibration_joint_names: tuple[str, ...] = (
        "l3b_femorotibial_back",
        "l4b_intertarsal_back",
        "l3f_femorotibial_front",
        "l4f_intertarsal_front",
        "l4p_intertarsal_pulley",
        "l5_metatarsophalangeal",
        "l6_interphalangeal",
        "l8_knee_flexor",
        "r3b_femorotibial_back",
        "r4b_intertarsal_back",
        "r3f_femorotibial_front",
        "r4f_intertarsal_front",
        "r4p_intertarsal_pulley",
        "r5_metatarsophalangeal",
        "r6_interphalangeal",
        "r8_knee_flexor",
    )


@dataclass
class PsoConfig:
    """Top-level PSO configuration."""

    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    parameters: list[ParameterSpec] = field(default_factory=list)

    @classmethod
    def default(cls) -> PsoConfig:
        cfg = cls()
        cfg.parameters = default_parameter_specs()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> PsoConfig:
        cfg = cls.default()
        if path is None:
            _validate_config(cfg)
            return cfg
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PSO config file does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}
        if not isinstance(values, dict):
            raise ValueError(f"PSO config must contain a mapping at top level: {path}")
        _merge_dataclass(cfg, values, path="pso")
        _validate_config(cfg)
        return cfg


def default_parameter_specs() -> list[ParameterSpec]:
    """Conservative Forrest PSO gait/preload search space.

    Nominal tendon path lengths are geometry/calibration parameters for the
    Forrest mechanism. PSO should not freely redesign them while tuning a gait;
    keep them in the Forrest parameter profile and only search GST spring rest
    length as the main tendon preload variable.
    """

    two_pi = 2.0 * math.pi
    return [
        ParameterSpec("tendons.baseline.lengths.gst_spring_rest", 0.035, 0.090, 0.060),
        ParameterSpec("tendons.baseline.lengths.upper_gst", 0.560, 0.700, 0.6217),
        ParameterSpec("tendons.baseline.lengths.lower_gst", 0.570, 0.710, 0.6314),
        ParameterSpec("tendons.baseline.lengths.dft", 0.320, 0.460, 0.384),
        ParameterSpec("tendons.baseline.lengths.edt1", 0.450, 0.630, 0.540),
        ParameterSpec("tendons.baseline.lengths.edt2", 0.520, 0.740, 0.630),
        ParameterSpec("run.cpg_oscillator.f_hz", 0.00, 5.00, 0.80),
        ParameterSpec("run.cpg_oscillator.duty_factor", 0.05, 0.95, 0.60),
        ParameterSpec("run.cpg_oscillator.right_phase_rad", 0.0, two_pi, math.pi),
        ParameterSpec("run.cpg_oscillator.hip_flexion_amplitude_deg", 0.0, 140.0, 24.0),
        ParameterSpec("run.cpg_oscillator.hip_flexion_offset_deg", -120.0, 120.0, 8.0),
        ParameterSpec("run.cpg_oscillator.hip_flexion_phase_rad", 0.0, two_pi, 0.0),
        ParameterSpec("run.cpg_oscillator.knee_flexion_amplitude_deg", 0.0, 180.0, 34.0),
        ParameterSpec("run.cpg_oscillator.knee_flexion_offset_deg", -120.0, 120.0, 0.0),
        ParameterSpec("run.cpg_oscillator.knee_flexion_phase_rad", 0.0, two_pi, math.pi / 2.0),
        ParameterSpec("run.cpg_oscillator.knee_swing_power", 0.05, 12.0, 1.5),
        ParameterSpec("run.cpg_oscillator.hip_roll_amplitude_deg", 0.0, 90.0, 0.0),
        ParameterSpec("run.cpg_oscillator.hip_roll_offset_deg", -60.0, 60.0, 0.0),
        ParameterSpec("run.cpg_oscillator.hip_roll_phase_rad", 0.0, two_pi, 0.0),
        ParameterSpec("run.cpg_oscillator.hip_yaw_amplitude_deg", 0.0, 90.0, 0.0),
        ParameterSpec("run.cpg_oscillator.hip_yaw_offset_deg", -60.0, 60.0, 0.0),
        ParameterSpec("run.cpg_oscillator.hip_yaw_phase_rad", 0.0, two_pi, 0.0),
    ]


def _merge_dataclass(target: Any, values: dict[str, Any], *, path: str) -> None:
    field_names = {field_.name for field_ in fields(target)}
    for key, value in values.items():
        if key not in field_names:
            raise ValueError(f"Unknown PSO parameter '{path}.{key}'")
        current = getattr(target, key)
        if key == "parameters":
            if not isinstance(value, list):
                raise ValueError(f"Expected '{path}.parameters' to be a list")
            setattr(target, key, [_parameter_spec_from_dict(item) for item in value])
        elif is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value, path=f"{path}.{key}")
        else:
            setattr(target, key, value)


def _parameter_spec_from_dict(value: dict[str, Any]) -> ParameterSpec:
    if not isinstance(value, dict):
        raise ValueError(f"Expected parameter spec to be a mapping, got: {value!r}")
    required = {"name", "min", "max"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Missing PSO parameter spec keys: {missing}")
    unknown = sorted(set(value) - {"name", "min", "max", "initial"})
    if unknown:
        raise ValueError(f"Unknown PSO parameter spec keys: {unknown}")
    return ParameterSpec(
        name=str(value["name"]),
        min=float(value["min"]),
        max=float(value["max"]),
        initial=None if value.get("initial") is None else float(value["initial"]),
    )


def _validate_config(cfg: PsoConfig) -> None:
    if int(cfg.swarm.num_particles) <= 0:
        raise ValueError("pso.swarm.num_particles must be positive.")
    if int(cfg.swarm.iterations) < 0:
        raise ValueError("pso.swarm.iterations must be non-negative.")
    if int(cfg.swarm.rollouts_per_iteration) <= 0:
        raise ValueError("pso.swarm.rollouts_per_iteration must be positive.")
    if cfg.objective.num_envs is not None and int(cfg.objective.num_envs) <= 0:
        raise ValueError("pso.objective.num_envs must be positive when set.")
    if float(cfg.objective.duration) <= 0.0:
        raise ValueError("pso.objective.duration must be positive.")
    if float(cfg.objective.startup_hold_duration) < 0.0:
        raise ValueError("pso.objective.startup_hold_duration must be non-negative.")
    command = tuple(float(value) for value in cfg.objective.command)
    if len(command) != 3:
        raise ValueError("pso.objective.command must contain exactly three values: [lin_x, lin_y, yaw].")
    cfg.objective.command = command
    if not cfg.parameters:
        raise ValueError("pso.parameters must contain at least one parameter.")
    for spec in cfg.parameters:
        if not spec.name:
            raise ValueError("pso.parameters entries must have non-empty names.")
        if not spec.max > spec.min:
            raise ValueError(f"PSO parameter max must be greater than min: {spec.name}")
        if spec.initial is not None and not spec.min <= spec.initial <= spec.max:
            raise ValueError(f"PSO parameter initial value is outside bounds: {spec.name}")
