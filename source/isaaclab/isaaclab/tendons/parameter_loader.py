# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized Forrest parameter loading.

The YAML file intentionally uses descriptive names instead of tensor indices.
This module owns the translation from debuggable config sections to the Isaac
Lab and tendon-model objects used at runtime.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from isaaclab.curriculums.command_bins import CommandBinCurriculumParameters

DEFAULT_FORREST_CONFIG_ENV = "ISAACNEXT_FORREST_CONFIG"
DEFAULT_FORREST_CONFIG_RELATIVE_PATH = Path("configs/forrest/default")
PROFILE_CONFIG_FILENAMES = ("base.yaml", "train.yaml", "run.yaml", "agent.yaml")

TENDON_NAMES = ("gst", "dft", "edt1", "edt2", "kft")
JOINT_INDEX = {
    "j3": 0,
    "j4": 1,
    "j5": 2,
    "j6": 3,
    "j8": 4,
}
PULLEY_RADIUS_INDEX = {
    "gst_3": 0,
    "gst_4": 1,
    "gst_4prime": 2,
    "gst_5": 3,
    "gst_6": 4,
    "dft_5": 5,
    "dft_6": 6,
    "edt1_5": 7,
    "edt2_5": 8,
    "edt2_6": 9,
    "kft_8": 10,
}
CHAIN_LINK_INDEX = {
    "chain_23": 0,
    "chain_34": 1,
    "chain_4prime5": 2,
    "chain_56": 3,
    "chain_67": 4,
    "chain_38": 5,
}
CONNECTOR_LINK_INDEX = {
    "gst_23": 0,
    "dft_c5": 1,
    "edt1_c4": 2,
    "edt1_5c": 3,
    "edt2_c4": 4,
    "kft_3c": 5,
}


def _range() -> tuple[float, float]:
    return (0.0, 0.0)


def _constant_range_dict(
    keys: tuple[str, ...] | list[str],
    value: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    return {key: value for key in keys}


@dataclass
class RobotAssetParameters:
    """Asset and spawn parameters for Forrest."""

    prim_path: str = "{ENV_REGEX_NS}/forrest_isaac"
    usd_path: str = "symlinks/forrest_ws/urdf/forrest_isaac/forrest_isaac.usd"
    height_scanner_prim_path: str = "{ENV_REGEX_NS}/forrest_isaac/base_assy_v2_1"
    initial_base_position: tuple[float, float, float] = (0.0, 0.0, 1.45)
    soft_joint_pos_limit_factor: float = 0.9
    fixed_world_body_path: str = "/World/Bot/base_assy_v2_1"
    fixed_world_joint_path: str = "/World/Bot/base_assy_v2_1_fixed_joint"
    fixed_world_joint_local_pos0: tuple[float, float, float] | None = None
    fixed_world_joint_local_rot0_wxyz: tuple[float, float, float, float] | None = None


@dataclass
class BoomParameters:
    """Planar boom constraint parameters for Forrest manager-based envs."""

    body_path_template: str = "/World/envs/env_{env_id}/forrest_isaac/base_assy_v2_1"
    joint_path_template: str = "/World/envs/env_{env_id}/forrest_isaac/base_assy_v2_1_planar_boom_joint"
    locked_axes: tuple[str, ...] = ("transY", "rotX", "rotZ")
    lock_x_angle: bool = False
    body_anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_anchor_rot_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    debug: bool = False


@dataclass
class RigidBodyPhysicsParameters:
    disable_gravity: bool = False
    retain_accelerations: bool = False
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    max_linear_velocity: float = 1000.0
    max_angular_velocity: float = 1000.0
    max_depenetration_velocity: float = 1.0


@dataclass
class ArticulationPhysicsParameters:
    enabled_self_collisions: bool = False
    solver_position_iteration_count: int = 4
    solver_velocity_iteration_count: int = 1
    selective_self_collision_body_names: tuple[str, ...] = (
        "s23_assy_1",
        "s34_foot_connector_assy_1",
        "s45_digit_assy_1",
        "s23_assy_2",
        "s34_foot_connector_assy_2",
        "s45_digit_assy_2",
    )
    selective_self_collision_debug: bool = False


@dataclass
class PhysicsParameters:
    sim_dt: float = 0.0024
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    virtual_ground_height: float | None = None
    physx_gpu_collision_stack_size: int = 160 * 1024 * 1024
    physx_gpu_found_lost_aggregate_pairs_capacity: int = 2**27
    physx_gpu_total_aggregate_pairs_capacity: int = 2**22
    rigid_body: RigidBodyPhysicsParameters = field(default_factory=RigidBodyPhysicsParameters)
    articulation: ArticulationPhysicsParameters = field(default_factory=ArticulationPhysicsParameters)


@dataclass
class ActuatorParameters:
    joint_names_expr: list[str] = field(default_factory=list)
    effort_limit_sim: float = 10.0e6
    velocity_limit_sim: float = 100.0
    stiffness: float = 500.0
    damping: float = 0.1


@dataclass
class ActuationParameters:
    flexor_angle_deg: float = 0.0
    pantograph: ActuatorParameters = field(
        default_factory=lambda: ActuatorParameters(
            joint_names_expr=["rp1_pantograph", "lp1_pantograph"],
            effort_limit_sim=1.0e6,
            velocity_limit_sim=100.0,
            stiffness=128.0e3,
            damping=0.1,
        )
    )
    hip_swing: ActuatorParameters = field(
        default_factory=lambda: ActuatorParameters(
            joint_names_expr=["r2_pseudo_acetabulofemoral_flexion", "l2_pseudo_acetabulofemoral_flexion"]
        )
    )
    hip_roll: ActuatorParameters = field(
        default_factory=lambda: ActuatorParameters(
            joint_names_expr=["r0_acetabulofemoral_roll", "l0_acetabulofemoral_roll"]
        )
    )
    hip_lateral: ActuatorParameters = field(
        default_factory=lambda: ActuatorParameters(
            joint_names_expr=["r1_acetabulofemoral_lateral", "l1_acetabulofemoral_lateral"]
        )
    )
    knee_flex: ActuatorParameters = field(
        default_factory=lambda: ActuatorParameters(joint_names_expr=["r8_knee_flexor", "l8_knee_flexor"])
    )
    passive_tendon_chain: ActuatorParameters = field(
        default_factory=lambda: ActuatorParameters(
            joint_names_expr=[
                "l3b_femorotibial_back",
                "l4b_intertarsal_back",
                "l3f_femorotibial_front",
                "l4f_intertarsal_front",
                "l4p_intertarsal_pulley",
                "l5_metatarsophalangeal",
                "l6_interphalangeal",
                "r3b_femorotibial_back",
                "r4b_intertarsal_back",
                "r3f_femorotibial_front",
                "r4f_intertarsal_front",
                "r4p_intertarsal_pulley",
                "r5_metatarsophalangeal",
                "r6_interphalangeal",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.05,
        )
    )


@dataclass
class TendonBaselineParameters:
    stiffness: dict[str, float] = field(
        default_factory=lambda: {
            "gst": 2.0e5,
            "dft": 5.0e4,
            "edt1": 5.0e5,
            "edt2": 5.0e5,
            "kft": 5.0e5,
        }
    )
    lengths: dict[str, float] = field(
        default_factory=lambda: {
            "gst_spring_rest": 0.06,
            "upper_gst": 0.6917,
            "lower_gst": 0.6314,
            "dft": 0.384,
            "edt1": 0.54,
            "edt2": 0.65,
            "kft": 0.452,
        }
    )
    joint_offsets_theta_deg: dict[str, float] = field(
        default_factory=lambda: {
            "j3": 227.671,
            "j4": 225.931,
            "j5": 180.0,
            "j6": 270.0,
            "j8": 180.0,
        }
    )
    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "j3": -1.0,
            "j4": 1.0,
            "j5": -1.0,
            "j6": -1.0,
            "j8": -1.0,
        }
    )
    pulley_radii: dict[str, float] = field(
        default_factory=lambda: {
            "gst_3": 0.029,
            "gst_4": 0.1,
            "gst_4prime": 0.05,
            "gst_5": 0.04,
            "gst_6": 0.01,
            "dft_5": 0.04,
            "dft_6": 0.01,
            "edt1_5": 0.04,
            "edt2_5": 0.04,
            "edt2_6": 0.01,
            "kft_8": 0.035,
        }
    )
    chain_link_lengths: dict[str, float] = field(
        default_factory=lambda: {
            "chain_23": 0.33,
            "chain_38": 0.33,
            "chain_34": 0.461,
            "chain_4prime5": 0.357,
            "chain_56": 0.165,
            "chain_67": 0.044,
        }
    )
    connector_link_lengths_longitudinal: dict[str, float] = field(
        default_factory=lambda: {
            "gst_23": 0.1072,
            "dft_c5": 0.13,
            "edt1_c4": 0.17,
            "edt1_5c": 0.088,
            "edt2_c4": 0.22,
            "kft_3c": 0.0635,
        }
    )
    connector_link_lengths_lateral: dict[str, float] = field(
        default_factory=lambda: {
            "gst_23": 0.0,
            "dft_c5": 0.04,
            "edt1_c4": 0.04,
            "edt1_5c": 0.007,
            "edt2_c4": 0.04,
            "kft_3c": 0.009,
        }
    )
    angles_deg: dict[str, float] = field(
        default_factory=lambda: {
            "gst_phi_23_j3": 98.874,
            "angle_4prime5_to_j44prime": 124.069,
        }
    )


@dataclass
class TendonRandomizationParameters:
    stiffness: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _constant_range_dict(TENDON_NAMES, _range())
    )
    lengths: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _constant_range_dict(
            ("gst_spring_rest", "upper_gst", "lower_gst", "dft", "edt1", "edt2", "kft"), _range()
        )
    )
    joint_offsets_theta: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _constant_range_dict(tuple(JOINT_INDEX), _range())
    )
    pulley_radii: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _constant_range_dict(tuple(PULLEY_RADIUS_INDEX), _range())
    )
    chain_link_lengths: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _constant_range_dict(tuple(CHAIN_LINK_INDEX), _range())
    )
    connector_link_lengths_longitudinal: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _constant_range_dict(tuple(CONNECTOR_LINK_INDEX), _range())
    )
    connector_link_lengths_lateral: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _constant_range_dict(tuple(CONNECTOR_LINK_INDEX), _range())
    )
    angles: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "gst_phi_23_j3": (0.0, 0.0),
            "angle_4prime5_to_j44prime": (0.0, 0.0),
        }
    )


@dataclass
class TendonParameters:
    baseline: TendonBaselineParameters = field(default_factory=TendonBaselineParameters)
    randomization: TendonRandomizationParameters = field(default_factory=TendonRandomizationParameters)
    damping: dict[str, float] = field(
        default_factory=lambda: {
            "gst": 20.0,
            "dft": 30.0,
            "edt1": 30.0,
            "edt2": 30.0,
            "kft": 30.0,
        }
    )


@dataclass
class RunCPGControllerParameters:
    f_hz: float = 1.5
    duty_factor: float = 0.60
    hip_amplitude_deg: float = 32.0
    hip_offset_deg: float = 22.0
    knee_amplitude_deg: float = 120.0
    swing_start_offset: float = 0.02
    swing_end_offset: float = 0.05
    combined_phase_offset_rad: float = math.pi / 2
    left_phase_offset_rad: float = -math.pi / 2
    right_phase_offset_rad: float = math.pi / 2
    include_knee: bool = True


@dataclass
class RunCpgOscillatorControllerParameters:
    f_hz: float = 0.8
    duty_factor: float = 0.60
    left_phase_rad: float = 0.0
    right_phase_rad: float = math.pi
    hip_flexion_amplitude_deg: float = 24.0
    hip_flexion_offset_deg: float = 8.0
    hip_flexion_phase_rad: float = 0.0
    knee_flexion_amplitude_deg: float = 34.0
    knee_flexion_offset_deg: float = 0.0
    knee_flexion_phase_rad: float = math.pi / 2.0
    knee_swing_power: float = 1.5
    hip_roll_amplitude_deg: float = 0.0
    hip_roll_offset_deg: float = 0.0
    hip_roll_phase_rad: float = 0.0
    hip_yaw_amplitude_deg: float = 0.0
    hip_yaw_offset_deg: float = 0.0
    hip_yaw_phase_rad: float = 0.0


@dataclass
class RunSinusoidalControllerParameters:
    f_hz: float = 3.0
    left_phi0_rad: float = 0.0
    right_phi0_rad: float = 0.0
    amplitude_deg: dict[str, float] = field(
        default_factory=lambda: {
            "hip_roll": 0.0,
            "hip_yaw": 0.0,
            "hip_flexion": 45.0,
            "knee_flexion": 75.0,
        }
    )
    offset_deg: dict[str, float] = field(
        default_factory=lambda: {
            "hip_roll": 0.0,
            "hip_yaw": 0.0,
            "hip_flexion": 0.0,
            "knee_flexion": -75.0,
        }
    )
    left_phase_rad: dict[str, float] = field(default_factory=lambda: {"hip_flexion": 0.0, "knee_flexion": 0.0})
    right_phase_rad: dict[str, float] = field(default_factory=lambda: {"hip_flexion": math.pi, "knee_flexion": math.pi})


@dataclass
class RunScriptParameters:
    duration: float = 2.0
    status_interval: int = 100
    startup_hold_enabled: bool = True
    startup_hold_duration: float = 2.0
    controller: str = "cpg"
    constraint_mode: str = "boom"
    cpg: RunCPGControllerParameters = field(default_factory=RunCPGControllerParameters)
    cpg_oscillator: RunCpgOscillatorControllerParameters = field(default_factory=RunCpgOscillatorControllerParameters)
    sinusoidal: RunSinusoidalControllerParameters = field(default_factory=RunSinusoidalControllerParameters)
    output_dir: str = "outputs"
    video_output: str = "outputs/simulation.mp4"


@dataclass
class AgentRunnerParameters:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50
    experiment_name: str = "forrest_rough"
    flat_experiment_name: str = "forrest_flat"
    run_name: str = ""
    logger: str = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    empirical_normalization: bool = False
    obs_groups: dict[str, list[str]] | None = None
    clip_actions: float | None = None
    check_for_nan: bool = True
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


@dataclass
class AgentPolicyParameters:
    init_noise_std: float = 1.6
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"


@dataclass
class AgentAlgorithmParameters:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.008
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adam"
    normalize_advantage_per_mini_batch: bool = False


@dataclass
class AgentParameters:
    runner: AgentRunnerParameters = field(default_factory=AgentRunnerParameters)
    policy: AgentPolicyParameters = field(default_factory=AgentPolicyParameters)
    algorithm: AgentAlgorithmParameters = field(default_factory=AgentAlgorithmParameters)


@dataclass
class RewardParameters:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "alive": 0.0,
            "termination_penalty": -200.0,
            "track_base_height_exp": 0.0,
            "track_lin_vel_xy_exp": 2.0,
            "forward_vel_x": 0.0,
            "track_ang_vel_z_exp": 0.1,
            "feet_crossing": -0.0,
            "feet_parallel_contact": -0.0,
            "feet_air_time": 0.0,
            "feet_slide": 0.0,
            "joint_deviation_l1": -0.1,
            "hip_deviation_l1": -0.3,
            "gait_symmetry": -0.1,
            "foot_connector_contact": -1.0,
            "lin_vel_z_l2": -0.1,
            "flat_orientation_l2": -1.0,
            "action_rate_l2": -0.0025,
            "dof_acc_l2": -1.25e-7,
            "dof_torques_l2": -1.5e-8,
        }
    )
    feet_crossing: dict[str, Any] = field(
        default_factory=lambda: {
            "min_lateral_separation": 0.10,
            "expected_foot0_lateral_order": -1.0,
            "side_margin": 0.02,
            "max_crossing_error": 0.25,
            "debug": False,
            "debug_every": 10,
            "debug_env_id": 0,
        }
    )
    feet_parallel_contact: dict[str, Any] = field(
        default_factory=lambda: {
            "contact_threshold": 1.0,
            "sole_normal_axis": (0.0, 0.0, 1.0),
            "ground_normal_w": (0.0, 0.0, 1.0),
            "debug": False,
            "debug_every": 100,
            "debug_env_id": 0,
        }
    )
    track_base_height: dict[str, float] = field(default_factory=lambda: {"target_height": 1.4, "std": 0.3})
    track_velocity: dict[str, float] = field(default_factory=lambda: {"lin_vel_xy_std": 0.5, "ang_vel_z_std": 0.5})
    feet_air_time_threshold: float = 0.4
    gait_symmetry: dict[str, Any] = field(
        default_factory=lambda: {
            "alpha": 0.001,
            "debug": False,
            "debug_every": 10,
            "debug_env_id": 0,
        }
    )
    undesired_contact_threshold: float = 1.0


@dataclass
class ActionParameters:
    scale_deg: dict[str, float] = field(
        default_factory=lambda: {
            ".*roll": 15.0,
            ".*lateral": 5.0,
            ".*flexion": 20.0,
            ".*flexor": 45.0,
        }
    )
    use_default_offset: bool = True


@dataclass
class ContactParameters:
    foot_body_names: tuple[str, str] = ("s45_digit_assy_1", "s45_digit_assy_2")
    foot_connector_body_names: tuple[str, str] = ("s34_foot_connector_assy_1", "s34_foot_connector_assy_2")
    contact_sensor_body_names: tuple[str, ...] = (
        "s45_digit_assy_1",
        "s45_digit_assy_2",
        "base_assy_v2_1",
        "outside_hip_v2_assy_axial_left_1",
        "outside_hip_v2_assy_axial_1",
        "differential_cage_assy_small_motor_1",
        "differential_cage_assy_small_motor_2",
        "s34_foot_connector_assy_1",
        "s34_foot_connector_assy_2",
    )
    base_termination_body_names: tuple[str, ...] = (
        "base_assy_v2_1",
        "differential_cage_assy_small_motor_1",
        "differential_cage_assy_small_motor_2",
        "outside_hip_v2_assy_axial_1",
        "outside_hip_v2_assy_axial_left_1",
    )
    update_period: float = 0.0
    history_length: int = 6
    debug_vis: bool = False
    track_air_time: bool = True
    right_foot_index: int = 0
    left_foot_index: int = 1
    forward_dir_b: tuple[float, float, float] = (1.0, 0.0, 0.0)
    lateral_dir_b: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass
class EventParameters:
    disable_push_robot: bool = True
    disable_add_base_mass: bool = True
    randomize_initial_base_pose: bool = True
    reset_robot_joint_position_range: tuple[float, float] = (1.0, 1.0)
    external_force_body_names: tuple[str, ...] = ("base_assy_v2_1",)
    base_com_body_names: tuple[str, ...] = ("base_assy_v2_1",)
    reset_base_pose_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.0, 0.0)}
    )
    reset_base_velocity_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )


@dataclass
class CommandParameters:
    lin_vel_x: tuple[float, float] = (2.0, 4.0)
    lin_vel_y: tuple[float, float] = (-0.0, 0.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)


@dataclass
class TerminationParameters:
    base_too_low_height: float = 0.2


@dataclass
class TrainingParameters:
    episode_length_s: float | None = None
    play_episode_length_s: float = 40.0
    rewards: RewardParameters = field(default_factory=RewardParameters)
    actions: ActionParameters = field(default_factory=ActionParameters)
    contacts: ContactParameters = field(default_factory=ContactParameters)
    events: EventParameters = field(default_factory=EventParameters)
    commands: CommandParameters = field(default_factory=CommandParameters)
    command_curriculum: CommandBinCurriculumParameters = field(default_factory=CommandBinCurriculumParameters)
    terminations: TerminationParameters = field(default_factory=TerminationParameters)


@dataclass
class ForrestParameterConfig:
    schema_version: int = 1
    robot: RobotAssetParameters = field(default_factory=RobotAssetParameters)
    boom: BoomParameters = field(default_factory=BoomParameters)
    physics: PhysicsParameters = field(default_factory=PhysicsParameters)
    actuation: ActuationParameters = field(default_factory=ActuationParameters)
    tendons: TendonParameters = field(default_factory=TendonParameters)
    run: RunScriptParameters = field(default_factory=RunScriptParameters)
    training: TrainingParameters = field(default_factory=TrainingParameters)
    agent: AgentParameters = field(default_factory=AgentParameters)

    @classmethod
    def from_yaml(cls, parameter_yaml_path: str | os.PathLike[str] | None = None) -> ForrestParameterConfig:
        """Load Forrest parameters from YAML.

        Missing fields use the typed defaults above. Unknown fields raise a
        clear error so typoed tuning parameters do not silently do nothing.
        """

        config = cls()
        path = resolve_forrest_config_path(parameter_yaml_path)
        if path is None:
            return config

        for file_path, parameters in iter_forrest_config_items(path):
            if not isinstance(parameters, dict):
                raise ValueError(f"Forrest parameter file must contain a mapping at top level: {file_path}")
            _merge_dataclass(config, parameters, path="forrest")
        return config

    def to_tendon_constants(self, *, device: Any | None = None):
        import torch

        from isaaclab.tendons.models.analytic.constants import (
            N_CHAIN_LINKS_PER_LEG,
            N_CONNECTOR_OFFSETS,
            N_JOINTS,
            N_RADII,
            TendonConstants,
            dev,
        )

        baseline = self.tendons.baseline
        constants = TendonConstants()
        target_device = dev if device is None else device

        for name in TENDON_NAMES:
            setattr(constants, f"{name}_stiffness", float(baseline.stiffness[name]))

        constants.gst_spring_rest_length = float(baseline.lengths["gst_spring_rest"])
        constants.upper_gst_length = float(baseline.lengths["upper_gst"])
        constants.lower_gst_length = float(baseline.lengths["lower_gst"])
        constants.dft_length = float(baseline.lengths["dft"])
        constants.edt1_length = float(baseline.lengths["edt1"])
        constants.edt2_length = float(baseline.lengths["edt2"])
        constants.kft_length = float(baseline.lengths["kft"])

        constants.joint_offsets_theta = torch.deg2rad(
            _named_tensor(baseline.joint_offsets_theta_deg, JOINT_INDEX, N_JOINTS, device=target_device)
        )
        constants.joint_directions = _named_tensor(
            baseline.joint_directions, JOINT_INDEX, N_JOINTS, device=target_device
        )
        constants.pulley_radii = _named_tensor(
            baseline.pulley_radii, PULLEY_RADIUS_INDEX, N_RADII, device=target_device
        )
        constants.chain_link_lengths = _named_tensor(
            baseline.chain_link_lengths, CHAIN_LINK_INDEX, N_CHAIN_LINKS_PER_LEG, device=target_device
        )
        constants.connector_link_lengths_longitudinal = _named_tensor(
            baseline.connector_link_lengths_longitudinal,
            CONNECTOR_LINK_INDEX,
            N_CONNECTOR_OFFSETS,
            device=target_device,
        )
        constants.connector_link_lengths_lateral = _named_tensor(
            baseline.connector_link_lengths_lateral,
            CONNECTOR_LINK_INDEX,
            N_CONNECTOR_OFFSETS,
            device=target_device,
        )
        constants.gst_phi_23_j3 = math.radians(float(baseline.angles_deg["gst_phi_23_j3"]))
        constants.angle_4prime5_to_j44prime = math.radians(float(baseline.angles_deg["angle_4prime5_to_j44prime"]))
        return constants

    def to_tendon_randomization_ranges(self):
        from isaaclab.tendons.models.analytic.constants import TendonConstantRandomizationRanges

        randomization = self.tendons.randomization
        return TendonConstantRandomizationRanges(
            gst_stiffness=_as_range(randomization.stiffness["gst"]),
            dft_stiffness=_as_range(randomization.stiffness["dft"]),
            edt1_stiffness=_as_range(randomization.stiffness["edt1"]),
            edt2_stiffness=_as_range(randomization.stiffness["edt2"]),
            kft_stiffness=_as_range(randomization.stiffness["kft"]),
            gst_spring_rest_length=_as_range(randomization.lengths["gst_spring_rest"]),
            upper_gst_length=_as_range(randomization.lengths["upper_gst"]),
            lower_gst_length=_as_range(randomization.lengths["lower_gst"]),
            dft_length=_as_range(randomization.lengths["dft"]),
            edt1_length=_as_range(randomization.lengths["edt1"]),
            edt2_length=_as_range(randomization.lengths["edt2"]),
            kft_length=_as_range(randomization.lengths["kft"]),
            joint_offsets_theta=_named_range_list(randomization.joint_offsets_theta, JOINT_INDEX, len(JOINT_INDEX)),
            pulley_radii=_named_range_list(randomization.pulley_radii, PULLEY_RADIUS_INDEX, len(PULLEY_RADIUS_INDEX)),
            chain_link_lengths=_named_range_list(
                randomization.chain_link_lengths, CHAIN_LINK_INDEX, len(CHAIN_LINK_INDEX)
            ),
            connector_link_lengths_longitudinal=_named_range_list(
                randomization.connector_link_lengths_longitudinal, CONNECTOR_LINK_INDEX, len(CONNECTOR_LINK_INDEX)
            ),
            connector_link_lengths_lateral=_named_range_list(
                randomization.connector_link_lengths_lateral, CONNECTOR_LINK_INDEX, len(CONNECTOR_LINK_INDEX)
            ),
            gst_phi_23_j3=_as_range(randomization.angles["gst_phi_23_j3"]),
            angle_4prime5_to_j44prime=_as_range(randomization.angles["angle_4prime5_to_j44prime"]),
        )

    def tendon_damping(self) -> dict[str, float]:
        return {name: _as_damping_value(value) for name, value in self.tendons.damping.items()}


def load_forrest_parameter_config(parameter_yaml_path: str | os.PathLike[str] | None = None) -> ForrestParameterConfig:
    return ForrestParameterConfig.from_yaml(parameter_yaml_path)


def iter_forrest_config_items(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    items = []
    for file_path in iter_forrest_config_files(path):
        items.extend(_load_forrest_config_file(file_path, visited=set()))
    return items


def iter_forrest_config_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = [path / filename for filename in PROFILE_CONFIG_FILENAMES]
        existing_files = [file_path for file_path in files if file_path.exists()]
        if not existing_files:
            raise FileNotFoundError(f"Forrest config directory contains none of {PROFILE_CONFIG_FILENAMES}: {path}")
        return existing_files
    return [path]


def _load_forrest_config_file(file_path: Path, *, visited: set[Path]) -> list[tuple[Path, dict[str, Any]]]:
    file_path = file_path.resolve()
    if file_path in visited:
        raise ValueError(f"Cyclic Forrest config include detected: {file_path}")
    visited.add(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file) or {}
    if not isinstance(parameters, dict):
        raise ValueError(f"Forrest parameter file must contain a mapping at top level: {file_path}")

    includes = parameters.pop("includes", [])
    if isinstance(includes, (str, os.PathLike)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(f"Forrest config 'includes' must be a string or list: {file_path}")

    items = []
    for include in includes:
        include_path = Path(include).expanduser()
        if not include_path.is_absolute():
            include_path = file_path.parent / include_path
        if include_path.is_dir():
            for include_file_path in iter_forrest_config_files(include_path):
                items.extend(_load_forrest_config_file(include_file_path, visited=visited))
        else:
            items.extend(_load_forrest_config_file(include_path, visited=visited))
    if parameters:
        items.append((file_path, parameters))
    return items


def resolve_forrest_config_path(parameter_yaml_path: str | os.PathLike[str] | None = None) -> Path | None:
    if parameter_yaml_path:
        path = Path(parameter_yaml_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Forrest parameter file or directory does not exist: {path}")
        return path

    env_path = os.environ.get(DEFAULT_FORREST_CONFIG_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{DEFAULT_FORREST_CONFIG_ENV} points to a missing file or directory: {path}")
        return path

    candidates = [Path.cwd() / DEFAULT_FORREST_CONFIG_RELATIVE_PATH]
    candidates.extend(parent / DEFAULT_FORREST_CONFIG_RELATIVE_PATH for parent in Path(__file__).resolve().parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _merge_dataclass(target: Any, values: dict[str, Any], *, path: str) -> None:
    field_names = {field_.name for field_ in fields(target)}
    for key, value in values.items():
        if key not in field_names:
            raise ValueError(f"Unknown Forrest parameter '{path}.{key}'")

        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value, path=f"{path}.{key}")
        elif isinstance(current, dict) and isinstance(value, dict):
            _merge_dict(current, value, path=f"{path}.{key}")
        else:
            setattr(target, key, value)


def _merge_dict(target: dict[str, Any], values: dict[str, Any], *, path: str) -> None:
    for key, value in values.items():
        if key not in target:
            raise ValueError(f"Unknown Forrest parameter '{path}.{key}'")
        target[key] = value


def _named_tensor(
    values: dict[str, float],
    index_map: dict[str, int],
    length: int,
    *,
    device: Any | None = None,
):
    import torch

    missing = sorted(set(index_map) - set(values))
    unknown = sorted(set(values) - set(index_map))
    if missing:
        raise ValueError(f"Missing named tendon constants: {missing}")
    if unknown:
        raise ValueError(f"Unknown named tendon constants: {unknown}")

    indexed = {index_map[name]: float(value) for name, value in values.items()}
    kwargs = {"dtype": torch.float32}
    if device is not None:
        kwargs["device"] = device
    return torch.tensor(_list_from_index_dict(indexed, length), **kwargs)


def _named_range_list(
    values: dict[str, tuple[float, float] | list[float]],
    index_map: dict[str, int],
    length: int,
) -> list[tuple[float, float]]:
    missing = sorted(set(index_map) - set(values))
    unknown = sorted(set(values) - set(index_map))
    if missing:
        raise ValueError(f"Missing named tendon randomization ranges: {missing}")
    if unknown:
        raise ValueError(f"Unknown named tendon randomization ranges: {unknown}")

    indexed = {index_map[name]: _as_range(value) for name, value in values.items()}
    return _list_from_index_dict(indexed, length)


def _list_from_index_dict(values: dict[int, Any], length: int) -> list[Any]:
    if set(values) != set(range(length)):
        raise ValueError(f"Expected consecutive indices 0..{length - 1}, got: {sorted(values)}")
    return [values[index] for index in range(length)]


def _as_range(value: tuple[float, float] | list[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"Expected a two-value range, got: {value}")
    return (float(value[0]), float(value[1]))


def _as_damping_value(value: float | tuple[float, float] | list[float]) -> float:
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or float(value[0]) != float(value[1]):
            raise ValueError(f"Tendon damping must be a scalar or fixed two-value range, got: {value}")
        return float(value[0])
    return float(value)
