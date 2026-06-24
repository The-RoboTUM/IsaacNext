# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched Forrest simulation evaluator for PSO particles."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.curriculums.command_bins import CommandBinCurriculumState, command_tracking_success
from isaaclab.pso.config import ObjectiveConfig
from isaaclab.pso.kernels import cpg_oscillator_command_kernel
from isaaclab.pso.parameters import ParameterSpace
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.tendons.manager import TendonManager
from isaaclab.tendons.models.analytic.tendon_data import TendonData
from isaaclab.tendons.parameter_loader import ForrestParameterConfig
from isaaclab.tendons.runner import (
    configure_scene_base_constraints,
    find_actuated_joint_indices,
    make_actuated_dof_specs,
    open_loop_command_batch,
    reset_robot_to_default,
    set_tendon_lengths_by_env,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from isaaclab_tasks.manager_based.locomotion.velocity.config.forrest.rl_env_cfg import (
    _feet_crossing_penalty_core,
    _feet_parallel_contact_penalty_core,
    _feet_symmetry_penalty_core,
    _track_base_height_exp_core,
    _unit_vec,
)

from isaaclab_assets.robots.forrest import get_forrest_cfg


@dataclass
class EvaluationResult:
    """Scores and rollout metrics for one PSO batch."""

    scores: torch.Tensor
    forward_speed: torch.Tensor
    forward_displacement: torch.Tensor
    lateral_displacement: torch.Tensor
    final_height: torch.Tensor
    fell: torch.Tensor
    unphysical: torch.Tensor
    backward: torch.Tensor
    terminated: torch.Tensor
    completed_rollouts: int
    fall_percent: float
    unphysical_percent: float
    backward_percent: float
    terminated_percent: float
    mean_survival_time: float
    mean_rollout_forward_speed: float
    max_rollout_forward_speed: float
    raw_max_rollout_forward_speed: float


def make_forrest_pso_scene_cfg(
    params: ForrestParameterConfig,
    *,
    num_envs: int,
    env_spacing: float,
    replicate_physics: bool,
    enable_contact_sensor: bool = True,
):
    """Create an InteractiveScene config for cloned Forrest evaluation envs."""

    robot_cfg = get_forrest_cfg(params).replace(prim_path="{ENV_REGEX_NS}/forrest_isaac")
    contact_params = params.training.contacts
    contact_body_regex = "(" + "|".join(contact_params.contact_sensor_body_names) + ")"

    @configclass
    class ForrestPsoSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        robot = robot_cfg
        if enable_contact_sensor:
            contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/forrest_isaac/" + contact_body_regex,
                update_period=float(contact_params.update_period),
                history_length=int(contact_params.history_length),
                debug_vis=bool(contact_params.debug_vis),
                track_air_time=bool(contact_params.track_air_time),
            )

    return ForrestPsoSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=replicate_physics,
    )


class ForrestPsoEvaluator:
    """Evaluate one swarm batch as a vectorized Isaac Lab rollout."""

    def __init__(
        self,
        *,
        forrest_params: ForrestParameterConfig,
        objective_cfg: ObjectiveConfig,
        parameter_space: ParameterSpace,
        device: str,
        num_particles: int,
        rollouts_per_iteration: int,
    ):
        self.forrest_params = forrest_params
        self.cfg = objective_cfg
        self.parameter_space = parameter_space
        self.num_particles = num_particles
        self.num_envs = int(objective_cfg.num_envs or num_particles)
        self.rollouts_per_iteration = max(int(rollouts_per_iteration), int(num_particles))
        self.reward_weights = self._reward_weights()
        self.enabled_reward_terms = {name for name, weight in self.reward_weights.items() if float(weight) != 0.0}
        self.enable_contact_sensor = bool(
            (
                {
                    "feet_parallel_contact",
                    "feet_air_time",
                    "feet_slide",
                    "foot_connector_contact",
                }
                & self.enabled_reward_terms
            )
            or bool(objective_cfg.terminations.terminate_on_undesired_contact)
        )

        sim_cfg = sim_utils.SimulationCfg(device=device, gravity=tuple(forrest_params.physics.gravity))
        sim_cfg.dt = (
            float(objective_cfg.sim_dt) if objective_cfg.sim_dt is not None else float(forrest_params.physics.sim_dt)
        )
        sim_cfg.physx.enable_external_forces_every_iteration = True
        sim_cfg.physx.min_velocity_iteration_count = max(1, int(sim_cfg.physx.min_velocity_iteration_count))
        sim_cfg.physx.gpu_collision_stack_size = int(forrest_params.physics.physx_gpu_collision_stack_size)
        sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = int(
            forrest_params.physics.physx_gpu_found_lost_aggregate_pairs_capacity
        )
        sim_cfg.physx.gpu_total_aggregate_pairs_capacity = int(
            forrest_params.physics.physx_gpu_total_aggregate_pairs_capacity
        )
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view([2.5, -8.0, 2.0], [2.5, 0.0, 0.85])

        replicate_physics = bool(objective_cfg.replicate_physics)
        if objective_cfg.constraint_mode in ("boom", "static") and replicate_physics:
            print(
                "[ForrestPsoEvaluator] Disabling replicate_physics because per-env base constraints are authored "
                "before PhysX startup."
            )
            replicate_physics = False

        scene_cfg = make_forrest_pso_scene_cfg(
            forrest_params,
            num_envs=self.num_envs,
            env_spacing=float(objective_cfg.env_spacing),
            replicate_physics=replicate_physics,
            enable_contact_sensor=self.enable_contact_sensor,
        )
        self.scene = InteractiveScene(scene_cfg)
        configure_scene_base_constraints(self.sim, forrest_params, objective_cfg.constraint_mode, self.num_envs)
        self.sim.reset()

        self.robot = self.scene["robot"]
        reset_robot_to_default(self.robot, env_origins=self.scene.env_origins)
        self.scene.reset()
        self.scene.update(0.0)

        self.actuated_dof_specs = make_actuated_dof_specs(scene_cfg.robot)
        self.actuated_joint_indices = find_actuated_joint_indices(
            self.robot, self.actuated_dof_specs, print_mapping=True
        )
        self.joint_side_ids = torch.tensor(
            [0 if spec.side == "left" else 1 for spec in self.actuated_dof_specs],
            dtype=torch.long,
            device=self.robot.device,
        )
        dof_ids = {"hip_roll": 0, "hip_yaw": 1, "hip_flexion": 2, "knee_flexion": 3}
        self.joint_dof_ids = torch.tensor(
            [dof_ids[spec.dof] for spec in self.actuated_dof_specs],
            dtype=torch.long,
            device=self.robot.device,
        )
        self.joint_signs = torch.tensor(
            [float(spec.sign) for spec in self.actuated_dof_specs],
            dtype=torch.float32,
            device=self.robot.device,
        )
        self.uses_cpg_oscillator = (
            any(name.startswith("run.cpg_oscillator.") for name in parameter_space.names)
            or forrest_params.run.controller == "cpg_oscillator"
        )

        self.tendon_data = TendonData(
            self.robot.num_instances,
            forrest_params.to_tendon_randomization_ranges(),
            tc=forrest_params.to_tendon_constants(device=self.robot.device),
            device=self.robot.device,
        )
        self.tendon_manager = TendonManager(
            self.robot,
            tendon_data=self.tendon_data,
            tendon_damping=forrest_params.tendon_damping(),
        )
        self.contact_sensor = self.scene.sensors.get("contact_forces")
        self._configure_reward_entities()
        self.command_curriculum = self._make_command_curriculum()

    def _configure_reward_entities(self) -> None:
        """Resolve body/joint ids used by RL-style reward terms."""

        contacts = self.forrest_params.training.contacts
        self.foot_body_indices, self.foot_body_names = self.robot.find_bodies(
            list(contacts.foot_body_names), preserve_order=True
        )
        self.foot_connector_body_indices, self.foot_connector_body_names = self.robot.find_bodies(
            list(contacts.foot_connector_body_names), preserve_order=True
        )
        if self.contact_sensor is not None:
            self.contact_foot_body_indices = [
                self.contact_sensor.body_names.index(name) for name in self.foot_body_names
            ]
            self.contact_foot_connector_body_indices = [
                self.contact_sensor.body_names.index(name) for name in self.foot_connector_body_names
            ]
            termination_cfg = self.cfg.terminations
            if termination_cfg.undesired_contact_body_names:
                undesired_body_names = tuple(termination_cfg.undesired_contact_body_names)
            else:
                foot_names = set(contacts.foot_body_names)
                undesired_body_names = tuple(
                    name for name in contacts.contact_sensor_body_names if name not in foot_names
                )
            missing_undesired = [name for name in undesired_body_names if name not in self.contact_sensor.body_names]
            if missing_undesired:
                raise ValueError(
                    "PSO undesired contact termination requested bodies that are not in the contact sensor: "
                    + ", ".join(missing_undesired)
                )
            self.undesired_contact_body_indices = [
                self.contact_sensor.body_names.index(name) for name in undesired_body_names
            ]
        else:
            self.contact_foot_body_indices = []
            self.contact_foot_connector_body_indices = []
            self.undesired_contact_body_indices = []
        self.hip_deviation_joint_indices, _ = self.robot.find_joints(
            [
                "l0_acetabulofemoral_roll",
                "r0_acetabulofemoral_roll",
            ],
            preserve_order=True,
        )
        self.joint_deviation_joint_indices, _ = self.robot.find_joints(
            [
                "l1_acetabulofemoral_lateral",
                "r1_acetabulofemoral_lateral",
            ],
            preserve_order=True,
        )
        self.dof_acc_joint_indices, _ = self.robot.find_joints(
            [
                "l0_acetabulofemoral_roll",
                "l1_acetabulofemoral_lateral",
                "l2_pseudo_acetabulofemoral_flexion",
                "r0_acetabulofemoral_roll",
                "r1_acetabulofemoral_lateral",
                "r2_pseudo_acetabulofemoral_flexion",
            ],
            preserve_order=True,
        )
        termination_cfg = self.cfg.terminations
        vibration_joint_names = tuple(termination_cfg.joint_vibration_joint_names)
        if vibration_joint_names:
            self.joint_vibration_joint_indices, _ = self.robot.find_joints(
                list(vibration_joint_names),
                preserve_order=True,
            )
        else:
            self.joint_vibration_joint_indices = []

    def _reward_weights(self) -> dict[str, float]:
        weights = dict(self.forrest_params.training.rewards.weights)
        weights.update({name: float(value) for name, value in self.cfg.reward_weights.items()})
        return weights

    def _rl_terminal_penalty(
        self,
        *,
        terminal: torch.Tensor,
        unphysical: torch.Tensor,
    ) -> torch.Tensor:
        """Return the weighted terminal penalty for RL-style PSO scoring."""

        penalties = torch.zeros_like(terminal, dtype=torch.float32)
        enabled = self.enabled_reward_terms
        if "termination_penalty" in enabled:
            penalties = penalties + terminal.to(dtype=torch.float32) * float(
                self.reward_weights.get("termination_penalty", 0.0)
            )
        if "unphysical_termination_penalty" in enabled:
            penalties = penalties + unphysical.to(dtype=torch.float32) * float(
                self.reward_weights.get("unphysical_termination_penalty", 0.0)
            )
        return penalties

    def _rl_reward_terms(
        self,
        *,
        command: torch.Tensor,
        terminal: torch.Tensor,
        previous_action: torch.Tensor,
        current_action: torch.Tensor,
        foot0_ahead_avg: torch.Tensor,
        foot1_ahead_avg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute weighted Forrest RL-style reward terms for PSO envs."""

        weights = self.reward_weights
        device = self.robot.device
        reward_params = self.forrest_params.training.rewards
        enabled = self.enabled_reward_terms
        next_foot0_avg = foot0_ahead_avg
        next_foot1_avg = foot1_ahead_avg

        terms: dict[str, torch.Tensor] = {}
        if "alive" in enabled:
            terms["alive"] = (~terminal).to(dtype=torch.float32)
        if "termination_penalty" in enabled:
            terms["termination_penalty"] = terminal.to(dtype=torch.float32)
        if "track_base_height_exp" in enabled:
            terms["track_base_height_exp"] = _track_base_height_exp_core(
                self.robot.data.root_pos_w[:, 2],
                float(reward_params.track_base_height["target_height"]),
                float(reward_params.track_base_height["std"]),
            )

        if "track_lin_vel_xy_exp" in enabled:
            vel_yaw = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w[:, :3])
            lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
            terms["track_lin_vel_xy_exp"] = torch.exp(
                -lin_vel_error / float(reward_params.track_velocity["lin_vel_xy_std"]) ** 2
            )
        if "forward_vel_x" in enabled:
            terms["forward_vel_x"] = torch.clamp(self.robot.data.root_lin_vel_w[:, 0], min=0.0)
        if "track_ang_vel_z_exp" in enabled:
            ang_vel_error = torch.square(command[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
            terms["track_ang_vel_z_exp"] = torch.exp(
                -ang_vel_error / float(reward_params.track_velocity["ang_vel_z_std"]) ** 2
            )

        needs_foot_pose = bool({"feet_crossing", "feet_parallel_contact", "gait_symmetry"} & enabled)
        if needs_foot_pose:
            foot_pos_b, foot_quat_w = self._feet_pose_base()
            if "feet_crossing" in enabled:
                terms["feet_crossing"] = _feet_crossing_penalty_core(
                    foot_pos_b,
                    _unit_vec(tuple(self.forrest_params.training.contacts.lateral_dir_b), device),
                    float(reward_params.feet_crossing["min_lateral_separation"]),
                    float(reward_params.feet_crossing["expected_foot0_lateral_order"]),
                    float(reward_params.feet_crossing["side_margin"]),
                    float(reward_params.feet_crossing["max_crossing_error"]),
                )
            if "feet_parallel_contact" in enabled:
                foot_forces = self.contact_sensor.data.net_forces_w[:, self.contact_foot_body_indices, :]
                terms["feet_parallel_contact"] = _feet_parallel_contact_penalty_core(
                    foot_quat_w,
                    foot_forces,
                    _unit_vec(tuple(reward_params.feet_parallel_contact["sole_normal_axis"]), device),
                    _unit_vec(tuple(reward_params.feet_parallel_contact["ground_normal_w"]), device),
                    float(reward_params.feet_parallel_contact["contact_threshold"]),
                )
            if "gait_symmetry" in enabled:
                forward_dir = _unit_vec(tuple(self.forrest_params.training.contacts.forward_dir_b), device)
                symmetry_penalty, next_foot0_avg, next_foot1_avg, _ = _feet_symmetry_penalty_core(
                    foot_pos_b,
                    forward_dir,
                    foot0_ahead_avg,
                    foot1_ahead_avg,
                    float(reward_params.gait_symmetry["alpha"]),
                )
                terms["gait_symmetry"] = symmetry_penalty

        if "feet_air_time" in enabled:
            air_time = self.contact_sensor.data.current_air_time[:, self.contact_foot_body_indices]
            contact_time = self.contact_sensor.data.current_contact_time[:, self.contact_foot_body_indices]
            in_contact = contact_time > 0.0
            in_mode_time = torch.where(in_contact, contact_time, air_time)
            single_stance = torch.sum(in_contact.int(), dim=1) == 1
            feet_air_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
            feet_air_time = torch.clamp(feet_air_time, max=float(reward_params.feet_air_time_threshold))
            feet_air_time *= torch.norm(command[:, :2], dim=1) > 0.1
            terms["feet_air_time"] = feet_air_time

        needs_contact_history = bool({"feet_slide", "foot_connector_contact"} & enabled)
        if needs_contact_history:
            contact_history = self.contact_sensor.data.net_forces_w_history
            if "feet_slide" in enabled:
                foot_contacts = (
                    contact_history[:, :, self.contact_foot_body_indices, :].norm(dim=-1).max(dim=1)[0] > 1.0
                )
                foot_vel_xy = self.robot.data.body_lin_vel_w[:, self.foot_body_indices, :2]
                terms["feet_slide"] = torch.sum(foot_vel_xy.norm(dim=-1) * foot_contacts, dim=1)
            if "foot_connector_contact" in enabled:
                connector_contacts = contact_history[:, :, self.contact_foot_connector_body_indices, :].norm(
                    dim=-1
                ).max(dim=1)[0] > float(reward_params.undesired_contact_threshold)
                terms["foot_connector_contact"] = torch.sum(connector_contacts, dim=1)

        if "joint_deviation_l1" in enabled:
            terms["joint_deviation_l1"] = torch.sum(
                torch.abs(
                    self.robot.data.joint_pos[:, self.joint_deviation_joint_indices]
                    - self.robot.data.default_joint_pos[:, self.joint_deviation_joint_indices]
                ),
                dim=1,
            )
        if "hip_deviation_l1" in enabled:
            terms["hip_deviation_l1"] = torch.sum(
                torch.abs(
                    self.robot.data.joint_pos[:, self.hip_deviation_joint_indices]
                    - self.robot.data.default_joint_pos[:, self.hip_deviation_joint_indices]
                ),
                dim=1,
            )
        if "lin_vel_z_l2" in enabled:
            terms["lin_vel_z_l2"] = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        if "flat_orientation_l2" in enabled:
            terms["flat_orientation_l2"] = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        if "action_rate_l2" in enabled:
            terms["action_rate_l2"] = torch.sum(torch.square(current_action - previous_action), dim=1)
        if "dof_acc_l2" in enabled:
            terms["dof_acc_l2"] = torch.sum(
                torch.square(self.robot.data.joint_acc[:, self.dof_acc_joint_indices]), dim=1
            )
        if "dof_torques_l2" in enabled:
            terms["dof_torques_l2"] = torch.sum(
                torch.square(self.robot.data.applied_torque[:, self.actuated_joint_indices]), dim=1
            )

        weighted = torch.zeros(self.num_envs, device=device)
        for name, value in terms.items():
            weight = float(weights.get(name, 0.0))
            if weight != 0.0:
                weighted = weighted + weight * value
        weighted = torch.where(torch.isfinite(weighted), weighted, torch.full_like(weighted, -1.0e9))
        return weighted, next_foot0_avg, next_foot1_avg

    def _feet_pose_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        foot_pos_w = self.robot.data.body_pos_w[:, self.foot_body_indices, :]
        foot_quat_w = self.robot.data.body_quat_w[:, self.foot_body_indices, :]
        rel = foot_pos_w - self.robot.data.root_pos_w[:, None, :]
        base_rot = self.robot.data.root_quat_w
        base_rot = base_rot / torch.norm(base_rot, dim=-1, keepdim=True).clamp(min=1e-8)
        from isaaclab_tasks.manager_based.locomotion.velocity.config.forrest.rl_env_cfg import quat_to_rot_matrix

        rot = quat_to_rot_matrix(base_rot)
        rel_b = torch.einsum("nij,nkj->nki", rot.transpose(1, 2), rel)
        return torch.nan_to_num(rel_b, nan=0.0, posinf=0.0, neginf=0.0), foot_quat_w

    def _make_command_curriculum(self) -> CommandBinCurriculumState | None:
        base_params = self.forrest_params.training.command_curriculum
        override_params = self.cfg.command_curriculum
        params = replace(
            base_params,
            enabled=bool(override_params.enabled),
            bins=override_params.bins,
            include_stand_bin=bool(override_params.include_stand_bin),
            lin_vel_x_min=float(override_params.lin_vel_x_min),
            lin_vel_x_max=float(override_params.lin_vel_x_max),
            lin_vel_x_bin_width=float(override_params.lin_vel_x_bin_width),
            lin_vel_y=float(override_params.lin_vel_y),
            ang_vel_z=float(override_params.ang_vel_z),
            initial_unlocked_bin=int(override_params.initial_unlocked_bin),
            successes_to_unlock=int(override_params.successes_to_unlock),
            min_attempts_to_unlock=int(override_params.min_attempts_to_unlock),
            min_success_rate_to_unlock=float(override_params.min_success_rate_to_unlock),
            max_attempts_to_track=int(override_params.max_attempts_to_track),
            success_velocity_tolerance=float(override_params.success_velocity_tolerance),
            success_yaw_rate_tolerance=float(override_params.success_yaw_rate_tolerance),
            success_min_survival_fraction=float(override_params.success_min_survival_fraction),
            sample_only_frontier=bool(override_params.sample_only_frontier),
            reset_counts_on_unlock=bool(override_params.reset_counts_on_unlock),
            command_name=str(override_params.command_name),
        )
        if not bool(self.cfg.use_command_curriculum) or not bool(params.enabled):
            return None
        return CommandBinCurriculumState(
            params,
            device=self.robot.device,
            prefer_newer_bins=bool(self.cfg.prefer_newer_command_bins),
            older_bin_probability_decay=float(self.cfg.older_command_bin_probability_decay),
        )

    def curriculum_summary(self) -> dict | None:
        if self.command_curriculum is None:
            return None
        return self.command_curriculum.summary()

    @property
    def sim_dt(self) -> float:
        return float(self.sim.get_physics_dt())

    def evaluate_async(
        self,
        *,
        optimizer,
        total_rollouts: int,
        report_interval: int,
        total_iterations: int,
    ):
        """Run asynchronous PSO rollouts and yield report-window results.

        Rollouts update the optimizer as soon as they finish. Environment slots
        are immediately recycled to the particles with the fewest scheduled
        rollouts, so long-running successful rollouts no longer block updates
        from faster completed rollouts.
        """

        total_rollouts = max(1, int(total_rollouts))
        report_interval = max(1, int(report_interval))
        num_steps = max(1, int(float(self.cfg.duration) / self.sim_dt))
        score_start_step = 0
        if self.cfg.eval_after_startup_hold:
            score_start_step = min(num_steps - 1, int(float(self.cfg.startup_hold_duration) / self.sim_dt))
        score_horizon_time = max(1, num_steps - score_start_step) * self.sim_dt
        termination_cfg = self.cfg.terminations
        backward_grace_steps = max(0, int(float(termination_cfg.backward_termination_grace_s) / self.sim_dt))
        undesired_contact_grace_steps = max(0, int(float(termination_cfg.undesired_contact_grace_s) / self.sim_dt))
        undesired_contact_consecutive_steps = max(1, int(termination_cfg.undesired_contact_consecutive_steps))
        joint_vibration_grace_steps = max(0, int(float(termination_cfg.joint_vibration_grace_s) / self.sim_dt))
        joint_vibration_consecutive_steps = max(1, int(termination_cfg.joint_vibration_consecutive_steps))

        device = self.robot.device
        dtype = torch.float32
        env_ids_all = torch.arange(self.num_envs, device=device, dtype=torch.long)
        active_envs = min(self.num_envs, total_rollouts)
        total_assigned = 0
        total_completed = 0
        report_completed = 0
        active_env_count = 0
        report_index = int(optimizer.iteration)
        improved_since_report = False

        completed_by_particle = torch.zeros(self.num_particles, dtype=torch.long, device=device)
        active_by_particle = torch.zeros(self.num_particles, dtype=torch.long, device=device)

        env_active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        env_particle = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        env_particle_position = torch.zeros(
            (self.num_envs, self.parameter_space.dim), dtype=torch.float32, device=device
        )
        rollout_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        scored_started = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        start_root_pos = torch.zeros((self.num_envs, 3), device=device)
        start_heading = torch.zeros(self.num_envs, device=device)
        initial_joint_positions = self.robot.data.default_joint_pos[:, self.actuated_joint_indices].clone()
        controller_zero = torch.zeros((self.num_envs, len(self.actuated_joint_indices)), device=device)
        rl_reward_integral = torch.zeros(self.num_envs, device=device)
        previous_action = torch.zeros((self.num_envs, len(self.actuated_joint_indices)), device=device)
        current_action = torch.zeros_like(previous_action)
        foot0_ahead_avg = torch.zeros(self.num_envs, device=device)
        foot1_ahead_avg = torch.zeros(self.num_envs, device=device)
        command_by_env = (
            torch.tensor(self.cfg.command, device=device, dtype=dtype).unsqueeze(0).repeat(self.num_envs, 1)
        )
        command_bin_by_env = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        undesired_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        joint_vibration_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        score_sum = torch.zeros(self.num_particles, device=device)
        rollout_count = torch.zeros(self.num_particles, device=device)
        forward_speed_sum = torch.zeros(self.num_particles, device=device)
        forward_displacement_sum = torch.zeros(self.num_particles, device=device)
        lateral_displacement_sum = torch.zeros(self.num_particles, device=device)
        final_height_sum = torch.zeros(self.num_particles, device=device)
        survival_time_sum = torch.zeros(self.num_particles, device=device)
        fall_count = torch.zeros(self.num_particles, device=device)
        unphysical_count = torch.zeros(self.num_particles, device=device)
        backward_count = torch.zeros(self.num_particles, device=device)
        terminated_count = torch.zeros(self.num_particles, device=device)
        rollout_forward_speed_sum = torch.zeros((), device=device)
        rollout_forward_speed_max = torch.tensor(-torch.inf, device=device)
        raw_rollout_forward_speed_max = torch.tensor(-torch.inf, device=device)

        current_params_by_env = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=device) for name in self.parameter_space.names
        }
        tendon_params_by_env = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            for name in self.parameter_space.names
            if name.startswith("tendons.baseline.lengths.")
        }
        cpg_params = self.forrest_params.run.cpg_oscillator
        cpg_tensors = {
            "run.cpg_oscillator.f_hz": torch.full((self.num_envs,), float(cpg_params.f_hz), device=device, dtype=dtype),
            "run.cpg_oscillator.duty_factor": torch.full(
                (self.num_envs,), float(cpg_params.duty_factor), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.left_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.left_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.right_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.right_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_flexion_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_flexion_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_flexion_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_flexion_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_flexion_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.hip_flexion_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_flexion_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.knee_flexion_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_flexion_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.knee_flexion_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_flexion_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.knee_flexion_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_swing_power": torch.full(
                (self.num_envs,), float(cpg_params.knee_swing_power), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_roll_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_roll_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_roll_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_roll_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_roll_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.hip_roll_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_yaw_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_yaw_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_yaw_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_yaw_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_yaw_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.hip_yaw_phase_rad), device=device, dtype=dtype
            ),
        }
        cpg_degree_params = {
            "run.cpg_oscillator.hip_flexion_amplitude_deg",
            "run.cpg_oscillator.hip_flexion_offset_deg",
            "run.cpg_oscillator.knee_flexion_amplitude_deg",
            "run.cpg_oscillator.knee_flexion_offset_deg",
            "run.cpg_oscillator.hip_roll_amplitude_deg",
            "run.cpg_oscillator.hip_roll_offset_deg",
            "run.cpg_oscillator.hip_yaw_amplitude_deg",
            "run.cpg_oscillator.hip_yaw_offset_deg",
        }

        def reset_report_accumulators() -> None:
            nonlocal rollout_forward_speed_sum, rollout_forward_speed_max, raw_rollout_forward_speed_max
            score_sum.zero_()
            rollout_count.zero_()
            forward_speed_sum.zero_()
            forward_displacement_sum.zero_()
            lateral_displacement_sum.zero_()
            final_height_sum.zero_()
            survival_time_sum.zero_()
            fall_count.zero_()
            unphysical_count.zero_()
            backward_count.zero_()
            terminated_count.zero_()
            rollout_forward_speed_sum = torch.zeros((), device=device)
            rollout_forward_speed_max = torch.tensor(-torch.inf, device=device)
            raw_rollout_forward_speed_max = torch.tensor(-torch.inf, device=device)

        def build_result(completed_rollouts: int) -> EvaluationResult:
            count = torch.clamp(rollout_count, min=1.0)
            scores = torch.where(rollout_count > 0, score_sum / count, torch.full_like(score_sum, -torch.inf))
            completed = max(1, int(completed_rollouts))
            mean_survival_time = float(
                (survival_time_sum.sum() / torch.clamp(rollout_count.sum(), min=1.0)).detach().cpu()
            )
            return EvaluationResult(
                scores=scores.detach(),
                forward_speed=(forward_speed_sum / count).detach(),
                forward_displacement=(forward_displacement_sum / count).detach(),
                lateral_displacement=(lateral_displacement_sum / count).detach(),
                final_height=(final_height_sum / count).detach(),
                fell=(fall_count > 0).detach(),
                unphysical=(unphysical_count > 0).detach(),
                backward=(backward_count > 0).detach(),
                terminated=(terminated_count > 0).detach(),
                completed_rollouts=int(completed_rollouts),
                fall_percent=float((fall_count.sum() / completed * 100.0).detach().cpu()),
                unphysical_percent=float((unphysical_count.sum() / completed * 100.0).detach().cpu()),
                backward_percent=float((backward_count.sum() / completed * 100.0).detach().cpu()),
                terminated_percent=float((terminated_count.sum() / completed * 100.0).detach().cpu()),
                mean_survival_time=mean_survival_time,
                mean_rollout_forward_speed=float((rollout_forward_speed_sum / completed).detach().cpu()),
                max_rollout_forward_speed=float(rollout_forward_speed_max.detach().cpu()),
                raw_max_rollout_forward_speed=float(raw_rollout_forward_speed_max.detach().cpu()),
            )

        def controller_command(
            *,
            t: float | torch.Tensor,
            initial_positions: torch.Tensor,
            zero: torch.Tensor,
        ) -> torch.Tensor:
            if self.uses_cpg_oscillator:
                t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
                if t_tensor.ndim == 0:
                    t_tensor = t_tensor.expand(self.num_envs)
                return cpg_oscillator_command_kernel(
                    t_tensor,
                    initial_positions,
                    zero,
                    self.joint_side_ids,
                    self.joint_dof_ids,
                    self.joint_signs,
                    cpg_tensors["run.cpg_oscillator.f_hz"],
                    cpg_tensors["run.cpg_oscillator.duty_factor"],
                    cpg_tensors["run.cpg_oscillator.left_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.right_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.hip_flexion_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_flexion_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_flexion_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.knee_flexion_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.knee_flexion_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.knee_flexion_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.knee_swing_power"],
                    cpg_tensors["run.cpg_oscillator.hip_roll_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_roll_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_roll_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.hip_yaw_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_yaw_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_yaw_phase_rad"],
                )
            return open_loop_command_batch(
                t=t,
                params_by_env=current_params_by_env,
                actuated_dof_specs=self.actuated_dof_specs,
                initial_joint_positions=initial_positions,
                controller_zero=zero,
            )

        def choose_particles(count: int) -> torch.Tensor:
            selected = []
            scheduled = completed_by_particle + active_by_particle
            remaining = int(count)
            while remaining > 0:
                take = min(remaining, self.num_particles)
                _values, ids = torch.topk(scheduled, k=take, largest=False)
                selected.append(ids)
                scheduled[ids] += 1
                remaining -= take
            return torch.cat(selected, dim=0)

        def assign_rollouts(env_ids: torch.Tensor) -> None:
            nonlocal active_env_count, total_assigned
            if env_ids.numel() == 0:
                return
            if total_assigned >= total_rollouts:
                reset_robot_to_default(self.robot, env_origins=self.scene.env_origins, env_ids=env_ids)
                self.tendon_manager.reset_damping_state(env_ids)
                env_active[env_ids] = False
                env_particle[env_ids] = -1
                rollout_step[env_ids] = 0
                scored_started[env_ids] = False
                undesired_contact_steps[env_ids] = 0
                joint_vibration_steps[env_ids] = 0
                return

            count = min(int(env_ids.numel()), total_rollouts - total_assigned)
            assign_env_ids = env_ids[:count]
            particle_ids = choose_particles(count)
            total_assigned += count
            active_by_particle.scatter_add_(0, particle_ids, torch.ones_like(particle_ids))

            reset_robot_to_default(self.robot, env_origins=self.scene.env_origins, env_ids=assign_env_ids)
            self.tendon_manager.reset_damping_state(assign_env_ids)

            particle_positions = optimizer.positions[particle_ids].detach().clone()
            physical_parameters = self.parameter_space.denormalize(particle_positions)
            env_particle_position[assign_env_ids] = particle_positions
            for param_index, name in enumerate(self.parameter_space.names):
                current_params_by_env[name][assign_env_ids] = physical_parameters[:, param_index]
                if name in tendon_params_by_env:
                    tendon_params_by_env[name][assign_env_ids] = physical_parameters[:, param_index]
                if self.uses_cpg_oscillator and name in cpg_tensors:
                    values = physical_parameters[:, param_index]
                    if name in cpg_degree_params:
                        values = torch.deg2rad(values)
                    cpg_tensors[name][assign_env_ids] = values
            selected_params = {name: values[assign_env_ids] for name, values in tendon_params_by_env.items()}
            set_tendon_lengths_by_env(self.tendon_data, selected_params, env_ids=assign_env_ids)

            env_active[assign_env_ids] = True
            active_env_count += count
            env_particle[assign_env_ids] = particle_ids
            rollout_step[assign_env_ids] = 0
            scored_started[assign_env_ids] = False
            start_root_pos[assign_env_ids] = self.robot.data.root_pos_w[assign_env_ids]
            start_heading[assign_env_ids] = self.robot.data.heading_w[assign_env_ids]
            initial_joint_positions[assign_env_ids] = self.robot.data.default_joint_pos[assign_env_ids][
                :, self.actuated_joint_indices
            ]
            zero_joint_positions = torch.zeros_like(initial_joint_positions)
            all_controller_zero = controller_command(
                t=0.0,
                initial_positions=zero_joint_positions,
                zero=zero_joint_positions,
            )
            controller_zero[assign_env_ids] = all_controller_zero[assign_env_ids]
            rl_reward_integral[assign_env_ids] = 0.0
            previous_action[assign_env_ids] = 0.0
            current_action[assign_env_ids] = 0.0
            foot0_ahead_avg[assign_env_ids] = 0.0
            foot1_ahead_avg[assign_env_ids] = 0.0
            undesired_contact_steps[assign_env_ids] = 0
            joint_vibration_steps[assign_env_ids] = 0
            if self.command_curriculum is not None:
                sampled_commands, sampled_bins = self.command_curriculum.sample(count)
                command_by_env[assign_env_ids] = sampled_commands
                command_bin_by_env[assign_env_ids] = sampled_bins
            else:
                command_by_env[assign_env_ids] = torch.tensor(self.cfg.command, device=device, dtype=dtype)
                command_bin_by_env[assign_env_ids] = 0

            if count < int(env_ids.numel()):
                env_active[env_ids[count:]] = False

        def finalize_rollouts(
            env_ids: torch.Tensor,
            *,
            fell: torch.Tensor,
            unphysical: torch.Tensor,
            backward: torch.Tensor,
            terminal: torch.Tensor,
        ) -> bool:
            nonlocal total_completed, report_completed, rollout_forward_speed_sum, rollout_forward_speed_max
            nonlocal raw_rollout_forward_speed_max, improved_since_report
            if env_ids.numel() == 0:
                return False

            particle_ids = env_particle[env_ids]
            final_root_pos = self.robot.data.root_pos_w[env_ids]
            scored = rollout_step[env_ids] > score_start_step
            displacement = final_root_pos - start_root_pos[env_ids]
            displacement = torch.where(scored.unsqueeze(1), displacement, torch.zeros_like(displacement))
            scored_steps = torch.clamp(rollout_step[env_ids] - score_start_step, min=1)
            evaluated_time = scored_steps.to(dtype=torch.float32) * self.sim_dt
            survival_time = rollout_step[env_ids].to(dtype=torch.float32) * self.sim_dt
            forward_displacement = displacement[:, 0]
            lateral_displacement = displacement[:, 1]
            raw_forward_speed = forward_displacement / evaluated_time
            failed = fell | unphysical | backward
            forward_speed = torch.where(failed, torch.zeros_like(raw_forward_speed), raw_forward_speed)
            reported_forward_speed = torch.clamp(
                forward_speed,
                min=-float(termination_cfg.max_forward_speed),
                max=float(termination_cfg.max_forward_speed),
            )
            final_height = final_root_pos[:, 2] - self.scene.env_origins[env_ids, 2]

            if self.command_curriculum is not None:
                heading_delta = wrap_to_pi(self.robot.data.heading_w[env_ids] - start_heading[env_ids])
                success_mask = command_tracking_success(
                    command_by_env[env_ids],
                    displacement[:, :2],
                    heading_delta,
                    evaluated_time,
                    terminal,
                    float(self.command_curriculum.params.success_min_survival_fraction) * score_horizon_time,
                    float(self.command_curriculum.params.success_velocity_tolerance),
                    float(self.command_curriculum.params.success_yaw_rate_tolerance),
                )
                self.command_curriculum.update(command_bin_by_env[env_ids], success_mask)

            score = rl_reward_integral[env_ids] / score_horizon_time + self._rl_terminal_penalty(
                terminal=terminal,
                unphysical=unphysical,
            )
            score = torch.where(torch.isfinite(score), score, torch.full_like(score, -1.0e9))

            improved = optimizer.observe_particles(particle_ids, env_particle_position[env_ids], score)
            optimizer.step_particles(particle_ids, total_iterations=total_iterations)
            improved_since_report = bool(improved_since_report or improved)

            ones = torch.ones_like(particle_ids)
            active_by_particle.scatter_add_(0, particle_ids, -ones)
            completed_by_particle.scatter_add_(0, particle_ids, ones)

            finite_forward_speed = torch.where(
                torch.isfinite(reported_forward_speed),
                reported_forward_speed,
                torch.full_like(reported_forward_speed, -torch.inf),
            )
            finite_raw_forward_speed = torch.where(
                torch.isfinite(raw_forward_speed),
                raw_forward_speed,
                torch.full_like(raw_forward_speed, -torch.inf),
            )
            rollout_forward_speed_sum = (
                rollout_forward_speed_sum
                + torch.where(
                    torch.isfinite(reported_forward_speed),
                    reported_forward_speed,
                    torch.zeros_like(reported_forward_speed),
                ).sum()
            )
            rollout_forward_speed_max = torch.maximum(rollout_forward_speed_max, finite_forward_speed.max())
            raw_rollout_forward_speed_max = torch.maximum(
                raw_rollout_forward_speed_max,
                finite_raw_forward_speed.max(),
            )

            score_sum.scatter_add_(0, particle_ids, score)
            rollout_count.scatter_add_(0, particle_ids, torch.ones_like(score))
            forward_speed_sum.scatter_add_(0, particle_ids, reported_forward_speed)
            forward_displacement_sum.scatter_add_(0, particle_ids, forward_displacement)
            lateral_displacement_sum.scatter_add_(0, particle_ids, lateral_displacement)
            final_height_sum.scatter_add_(0, particle_ids, final_height)
            survival_time_sum.scatter_add_(0, particle_ids, survival_time)
            unphysical_primary = terminal & unphysical
            fall_primary = terminal & (~unphysical_primary) & fell
            backward_primary = terminal & (~unphysical_primary) & (~fall_primary) & backward
            fall_count.scatter_add_(0, particle_ids, fall_primary.to(dtype=torch.float32))
            unphysical_count.scatter_add_(0, particle_ids, unphysical_primary.to(dtype=torch.float32))
            backward_count.scatter_add_(0, particle_ids, backward_primary.to(dtype=torch.float32))
            terminated_count.scatter_add_(0, particle_ids, terminal.to(dtype=torch.float32))
            total_completed += int(env_ids.numel())
            report_completed += int(env_ids.numel())
            return improved

        assign_rollouts(env_ids_all[:active_envs])

        while total_completed < total_rollouts and active_env_count > 0:
            active_ids = torch.nonzero(env_active, as_tuple=False).flatten()

            self.tendon_manager.apply_jit(
                virtual_ground_height=self.forrest_params.physics.virtual_ground_height,
                dt=self.sim_dt,
            )

            rollout_t = rollout_step.to(dtype=torch.float32) * self.sim_dt
            controller_t = torch.clamp(rollout_t - float(self.cfg.startup_hold_duration), min=0.0)
            active_commanded = controller_command(
                t=controller_t,
                initial_positions=initial_joint_positions,
                zero=controller_zero,
            )
            hold_mask = (rollout_t < float(self.cfg.startup_hold_duration)).unsqueeze(1)
            commanded_positions = torch.where(hold_mask, initial_joint_positions, active_commanded)
            previous_action[active_ids] = current_action[active_ids]
            current_action[active_ids] = commanded_positions[active_ids] - initial_joint_positions[active_ids]
            self.robot.set_joint_position_target(
                commanded_positions[active_ids],
                joint_ids=self.actuated_joint_indices,
                env_ids=active_ids,
            )

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

            rollout_step[active_ids] += 1
            just_started = env_active & (~scored_started) & (rollout_step >= score_start_step)
            if torch.any(just_started):
                start_root_pos[just_started] = self.robot.data.root_pos_w[just_started]
                start_heading[just_started] = self.robot.data.heading_w[just_started]
                scored_started[just_started] = True

            scoring_active = active_ids[scored_started[active_ids]]
            if scoring_active.numel() > 0:
                terminal_reward_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
                reward, foot0_ahead_avg, foot1_ahead_avg = self._rl_reward_terms(
                    command=command_by_env,
                    terminal=terminal_reward_mask,
                    previous_action=previous_action,
                    current_action=current_action,
                    foot0_ahead_avg=foot0_ahead_avg,
                    foot1_ahead_avg=foot1_ahead_avg,
                )
                rl_reward_integral[scoring_active] += reward[scoring_active] * self.sim_dt

            root_pos = self.robot.data.root_pos_w
            root_lin_vel = self.robot.data.root_lin_vel_w
            root_ang_vel = self.robot.data.root_ang_vel_w
            relative_height = root_pos[:, 2] - self.scene.env_origins[:, 2]
            displacement = root_pos - start_root_pos

            fell_now = relative_height < float(termination_cfg.base_too_low_height)
            if termination_cfg.terminate_on_backward_progress:
                backward_allowed_to_trip = scored_started & (rollout_step >= score_start_step + backward_grace_steps)
                backward_now = backward_allowed_to_trip & (
                    displacement[:, 0] < -float(termination_cfg.max_backward_displacement)
                )
            else:
                backward_now = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

            if termination_cfg.terminate_on_unphysical:
                finite_state = (
                    torch.isfinite(root_pos).all(dim=1)
                    & torch.isfinite(root_lin_vel).all(dim=1)
                    & torch.isfinite(root_ang_vel).all(dim=1)
                )
                unphysical_now = (
                    (~finite_state)
                    | (root_lin_vel[:, 0].abs() > float(termination_cfg.max_forward_speed))
                    | (root_lin_vel[:, 1].abs() > float(termination_cfg.max_lateral_speed))
                    | (root_lin_vel[:, 2].abs() > float(termination_cfg.max_vertical_speed))
                    | (torch.linalg.norm(root_ang_vel, dim=1) > float(termination_cfg.max_root_angular_speed))
                    | (relative_height > float(termination_cfg.max_height))
                )
                if (
                    termination_cfg.terminate_on_undesired_contact
                    and self.contact_sensor is not None
                    and len(self.undesired_contact_body_indices) > 0
                ):
                    contact_allowed_to_trip = rollout_step >= undesired_contact_grace_steps
                    contact_history = self.contact_sensor.data.net_forces_w_history
                    undesired_contact = contact_history[:, :, self.undesired_contact_body_indices, :].norm(dim=-1).amax(
                        dim=(1, 2)
                    ) > float(termination_cfg.undesired_contact_force_threshold)
                    contact_step = contact_allowed_to_trip & undesired_contact
                    undesired_contact_steps[:] = torch.where(
                        env_active & contact_step,
                        undesired_contact_steps + 1,
                        torch.zeros_like(undesired_contact_steps),
                    )
                    unphysical_now = unphysical_now | (undesired_contact_steps >= undesired_contact_consecutive_steps)
                if termination_cfg.terminate_on_joint_vibration and len(self.joint_vibration_joint_indices) > 0:
                    vibration_allowed_to_trip = rollout_step >= joint_vibration_grace_steps
                    joint_vel = self.robot.data.joint_vel[:, self.joint_vibration_joint_indices]
                    joint_acc = self.robot.data.joint_acc[:, self.joint_vibration_joint_indices]
                    finite_joints = torch.isfinite(joint_vel).all(dim=1) & torch.isfinite(joint_acc).all(dim=1)
                    vibration_step = vibration_allowed_to_trip & (
                        (~finite_joints)
                        | (joint_vel.abs().amax(dim=1) > float(termination_cfg.max_joint_vibration_velocity))
                        | (joint_acc.abs().amax(dim=1) > float(termination_cfg.max_joint_vibration_acceleration))
                    )
                    joint_vibration_steps[:] = torch.where(
                        env_active & vibration_step,
                        joint_vibration_steps + 1,
                        torch.zeros_like(joint_vibration_steps),
                    )
                    unphysical_now = unphysical_now | (joint_vibration_steps >= joint_vibration_consecutive_steps)
            else:
                unphysical_now = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

            reached_duration = rollout_step >= num_steps
            done = env_active & (fell_now | unphysical_now | backward_now | reached_duration)
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if done_ids.numel() == 0:
                continue

            finalize_rollouts(
                done_ids,
                fell=fell_now[done_ids],
                unphysical=unphysical_now[done_ids],
                backward=backward_now[done_ids],
                terminal=(fell_now | unphysical_now | backward_now)[done_ids],
            )
            env_active[done_ids] = False
            active_env_count -= int(done_ids.numel())
            assign_rollouts(done_ids)

            if report_completed >= report_interval or total_completed >= total_rollouts:
                result = build_result(report_completed)
                report_completed = 0
                report_improved = improved_since_report
                improved_since_report = False
                yield report_index, total_completed, report_improved, result
                report_index += 1
                reset_report_accumulators()

    def evaluate(self, physical_parameters: torch.Tensor) -> EvaluationResult:
        """Run scheduled rollouts and return mean maximization scores per particle."""

        physical_parameters = physical_parameters.to(device=self.robot.device, dtype=torch.float32)
        total_rollouts = self.rollouts_per_iteration
        active_envs = min(self.num_envs, total_rollouts)
        num_steps = max(1, int(float(self.cfg.duration) / self.sim_dt))
        score_start_step = 0
        if self.cfg.eval_after_startup_hold:
            score_start_step = min(num_steps - 1, int(float(self.cfg.startup_hold_duration) / self.sim_dt))
        score_horizon_time = max(1, num_steps - score_start_step) * self.sim_dt
        termination_cfg = self.cfg.terminations
        backward_grace_steps = max(0, int(float(termination_cfg.backward_termination_grace_s) / self.sim_dt))
        undesired_contact_grace_steps = max(0, int(float(termination_cfg.undesired_contact_grace_s) / self.sim_dt))
        undesired_contact_consecutive_steps = max(1, int(termination_cfg.undesired_contact_consecutive_steps))
        joint_vibration_grace_steps = max(0, int(float(termination_cfg.joint_vibration_grace_s) / self.sim_dt))
        joint_vibration_consecutive_steps = max(1, int(termination_cfg.joint_vibration_consecutive_steps))

        device = self.robot.device
        dtype = torch.float32
        env_ids_all = torch.arange(self.num_envs, device=device, dtype=torch.long)
        rollout_queue = torch.arange(total_rollouts, device=device, dtype=torch.long) % self.num_particles
        queue_index = 0
        completed_rollouts = 0
        active_env_count = 0

        score_sum = torch.zeros(self.num_particles, device=device)
        rollout_count = torch.zeros(self.num_particles, device=device)
        forward_speed_sum = torch.zeros(self.num_particles, device=device)
        forward_displacement_sum = torch.zeros(self.num_particles, device=device)
        lateral_displacement_sum = torch.zeros(self.num_particles, device=device)
        final_height_sum = torch.zeros(self.num_particles, device=device)
        survival_time_sum = torch.zeros(self.num_particles, device=device)
        fall_count = torch.zeros(self.num_particles, device=device)
        unphysical_count = torch.zeros(self.num_particles, device=device)
        backward_count = torch.zeros(self.num_particles, device=device)
        terminated_count = torch.zeros(self.num_particles, device=device)
        rollout_forward_speed_sum = torch.zeros((), device=device)
        rollout_forward_speed_max = torch.tensor(-torch.inf, device=device)
        raw_rollout_forward_speed_max = torch.tensor(-torch.inf, device=device)

        env_active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        env_particle = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        rollout_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        scored_started = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        start_root_pos = torch.zeros((self.num_envs, 3), device=device)
        start_heading = torch.zeros(self.num_envs, device=device)
        initial_joint_positions = self.robot.data.default_joint_pos[:, self.actuated_joint_indices].clone()
        controller_zero = torch.zeros((self.num_envs, len(self.actuated_joint_indices)), device=device)
        rl_reward_integral = torch.zeros(self.num_envs, device=device)
        previous_action = torch.zeros((self.num_envs, len(self.actuated_joint_indices)), device=device)
        current_action = torch.zeros_like(previous_action)
        foot0_ahead_avg = torch.zeros(self.num_envs, device=device)
        foot1_ahead_avg = torch.zeros(self.num_envs, device=device)
        command_by_env = (
            torch.tensor(self.cfg.command, device=device, dtype=dtype).unsqueeze(0).repeat(self.num_envs, 1)
        )
        command_bin_by_env = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        undesired_contact_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        joint_vibration_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        current_params_by_env = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=device) for name in self.parameter_space.names
        }
        tendon_params_by_env = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            for name in self.parameter_space.names
            if name.startswith("tendons.baseline.lengths.")
        }
        cpg_params = self.forrest_params.run.cpg_oscillator
        cpg_tensors = {
            "run.cpg_oscillator.f_hz": torch.full((self.num_envs,), float(cpg_params.f_hz), device=device, dtype=dtype),
            "run.cpg_oscillator.duty_factor": torch.full(
                (self.num_envs,), float(cpg_params.duty_factor), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.left_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.left_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.right_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.right_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_flexion_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_flexion_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_flexion_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_flexion_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_flexion_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.hip_flexion_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_flexion_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.knee_flexion_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_flexion_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.knee_flexion_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_flexion_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.knee_flexion_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.knee_swing_power": torch.full(
                (self.num_envs,), float(cpg_params.knee_swing_power), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_roll_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_roll_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_roll_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_roll_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_roll_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.hip_roll_phase_rad), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_yaw_amplitude_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_yaw_amplitude_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_yaw_offset_deg": torch.full(
                (self.num_envs,), math.radians(float(cpg_params.hip_yaw_offset_deg)), device=device, dtype=dtype
            ),
            "run.cpg_oscillator.hip_yaw_phase_rad": torch.full(
                (self.num_envs,), float(cpg_params.hip_yaw_phase_rad), device=device, dtype=dtype
            ),
        }
        cpg_degree_params = {
            "run.cpg_oscillator.hip_flexion_amplitude_deg",
            "run.cpg_oscillator.hip_flexion_offset_deg",
            "run.cpg_oscillator.knee_flexion_amplitude_deg",
            "run.cpg_oscillator.knee_flexion_offset_deg",
            "run.cpg_oscillator.hip_roll_amplitude_deg",
            "run.cpg_oscillator.hip_roll_offset_deg",
            "run.cpg_oscillator.hip_yaw_amplitude_deg",
            "run.cpg_oscillator.hip_yaw_offset_deg",
        }

        def controller_command(
            *,
            t: float | torch.Tensor,
            initial_positions: torch.Tensor,
            zero: torch.Tensor,
        ) -> torch.Tensor:
            if self.uses_cpg_oscillator:
                t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
                if t_tensor.ndim == 0:
                    t_tensor = t_tensor.expand(self.num_envs)
                return cpg_oscillator_command_kernel(
                    t_tensor,
                    initial_positions,
                    zero,
                    self.joint_side_ids,
                    self.joint_dof_ids,
                    self.joint_signs,
                    cpg_tensors["run.cpg_oscillator.f_hz"],
                    cpg_tensors["run.cpg_oscillator.duty_factor"],
                    cpg_tensors["run.cpg_oscillator.left_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.right_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.hip_flexion_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_flexion_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_flexion_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.knee_flexion_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.knee_flexion_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.knee_flexion_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.knee_swing_power"],
                    cpg_tensors["run.cpg_oscillator.hip_roll_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_roll_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_roll_phase_rad"],
                    cpg_tensors["run.cpg_oscillator.hip_yaw_amplitude_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_yaw_offset_deg"],
                    cpg_tensors["run.cpg_oscillator.hip_yaw_phase_rad"],
                )
            return open_loop_command_batch(
                t=t,
                params_by_env=current_params_by_env,
                actuated_dof_specs=self.actuated_dof_specs,
                initial_joint_positions=initial_positions,
                controller_zero=zero,
            )

        def assign_rollouts(env_ids: torch.Tensor) -> None:
            nonlocal active_env_count, queue_index
            if env_ids.numel() == 0 or queue_index >= total_rollouts:
                if env_ids.numel() > 0:
                    reset_robot_to_default(self.robot, env_origins=self.scene.env_origins, env_ids=env_ids)
                    self.tendon_manager.reset_damping_state(env_ids)
                    env_active[env_ids] = False
                    env_particle[env_ids] = -1
                    rollout_step[env_ids] = 0
                    scored_started[env_ids] = False
                    start_root_pos[env_ids] = (
                        self.robot.data.default_root_state[env_ids, :3] + self.scene.env_origins[env_ids]
                    )
                    start_heading[env_ids] = 0.0
                    initial_joint_positions[env_ids] = self.robot.data.default_joint_pos[env_ids][
                        :, self.actuated_joint_indices
                    ]
                    controller_zero[env_ids] = 0.0
                    rl_reward_integral[env_ids] = 0.0
                    previous_action[env_ids] = 0.0
                    current_action[env_ids] = 0.0
                    foot0_ahead_avg[env_ids] = 0.0
                    foot1_ahead_avg[env_ids] = 0.0
                    command_by_env[env_ids] = torch.tensor(self.cfg.command, device=device, dtype=dtype)
                    command_bin_by_env[env_ids] = 0
                    undesired_contact_steps[env_ids] = 0
                    joint_vibration_steps[env_ids] = 0
                return

            count = min(int(env_ids.numel()), total_rollouts - queue_index)
            assign_env_ids = env_ids[:count]
            particle_ids = rollout_queue[queue_index : queue_index + count]
            queue_index += count

            reset_robot_to_default(self.robot, env_origins=self.scene.env_origins, env_ids=assign_env_ids)
            self.tendon_manager.reset_damping_state(assign_env_ids)

            for param_index, name in enumerate(self.parameter_space.names):
                current_params_by_env[name][assign_env_ids] = physical_parameters[particle_ids, param_index]
                if name in tendon_params_by_env:
                    tendon_params_by_env[name][assign_env_ids] = physical_parameters[particle_ids, param_index]
                if self.uses_cpg_oscillator and name in cpg_tensors:
                    values = physical_parameters[particle_ids, param_index]
                    if name in cpg_degree_params:
                        values = torch.deg2rad(values)
                    cpg_tensors[name][assign_env_ids] = values
            selected_params = {name: values[assign_env_ids] for name, values in tendon_params_by_env.items()}
            set_tendon_lengths_by_env(self.tendon_data, selected_params, env_ids=assign_env_ids)

            env_active[assign_env_ids] = True
            active_env_count += count
            env_particle[assign_env_ids] = particle_ids
            rollout_step[assign_env_ids] = 0
            scored_started[assign_env_ids] = False
            start_root_pos[assign_env_ids] = self.robot.data.root_pos_w[assign_env_ids]
            start_heading[assign_env_ids] = self.robot.data.heading_w[assign_env_ids]
            initial_joint_positions[assign_env_ids] = self.robot.data.default_joint_pos[assign_env_ids][
                :, self.actuated_joint_indices
            ]
            zero_joint_positions = torch.zeros_like(initial_joint_positions)
            all_controller_zero = controller_command(
                t=0.0,
                initial_positions=zero_joint_positions,
                zero=zero_joint_positions,
            )
            controller_zero[assign_env_ids] = all_controller_zero[assign_env_ids]
            rl_reward_integral[assign_env_ids] = 0.0
            previous_action[assign_env_ids] = 0.0
            current_action[assign_env_ids] = 0.0
            foot0_ahead_avg[assign_env_ids] = 0.0
            foot1_ahead_avg[assign_env_ids] = 0.0
            undesired_contact_steps[assign_env_ids] = 0
            joint_vibration_steps[assign_env_ids] = 0
            if self.command_curriculum is not None:
                sampled_commands, sampled_bins = self.command_curriculum.sample(count)
                command_by_env[assign_env_ids] = sampled_commands
                command_bin_by_env[assign_env_ids] = sampled_bins
            else:
                command_by_env[assign_env_ids] = torch.tensor(self.cfg.command, device=device, dtype=dtype)
                command_bin_by_env[assign_env_ids] = 0

            if count < int(env_ids.numel()):
                env_active[env_ids[count:]] = False

        def finalize_rollouts(
            env_ids: torch.Tensor,
            *,
            fell: torch.Tensor,
            unphysical: torch.Tensor,
            backward: torch.Tensor,
            terminal: torch.Tensor,
        ) -> None:
            nonlocal completed_rollouts, rollout_forward_speed_sum, rollout_forward_speed_max
            nonlocal raw_rollout_forward_speed_max
            if env_ids.numel() == 0:
                return

            particle_ids = env_particle[env_ids]
            final_root_pos = self.robot.data.root_pos_w[env_ids]
            scored = rollout_step[env_ids] > score_start_step
            displacement = final_root_pos - start_root_pos[env_ids]
            displacement = torch.where(scored.unsqueeze(1), displacement, torch.zeros_like(displacement))
            scored_steps = torch.clamp(rollout_step[env_ids] - score_start_step, min=1)
            evaluated_time = scored_steps.to(dtype=torch.float32) * self.sim_dt
            survival_time = rollout_step[env_ids].to(dtype=torch.float32) * self.sim_dt
            forward_displacement = displacement[:, 0]
            lateral_displacement = displacement[:, 1]
            raw_forward_speed = forward_displacement / evaluated_time
            failed = fell | unphysical | backward
            forward_speed = torch.where(failed, torch.zeros_like(raw_forward_speed), raw_forward_speed)
            reported_forward_speed = torch.clamp(
                forward_speed,
                min=-float(termination_cfg.max_forward_speed),
                max=float(termination_cfg.max_forward_speed),
            )
            final_height = final_root_pos[:, 2] - self.scene.env_origins[env_ids, 2]

            if self.command_curriculum is not None:
                heading_delta = wrap_to_pi(self.robot.data.heading_w[env_ids] - start_heading[env_ids])
                success_mask = command_tracking_success(
                    command_by_env[env_ids],
                    displacement[:, :2],
                    heading_delta,
                    evaluated_time,
                    terminal,
                    float(self.command_curriculum.params.success_min_survival_fraction) * score_horizon_time,
                    float(self.command_curriculum.params.success_velocity_tolerance),
                    float(self.command_curriculum.params.success_yaw_rate_tolerance),
                )
                self.command_curriculum.update(command_bin_by_env[env_ids], success_mask)

            terminal_penalty = self._rl_terminal_penalty(terminal=terminal, unphysical=unphysical)
            score = rl_reward_integral[env_ids] / score_horizon_time + terminal_penalty
            score = torch.where(torch.isfinite(score), score, torch.full_like(score, -1.0e9))

            finite_forward_speed = torch.where(
                torch.isfinite(reported_forward_speed),
                reported_forward_speed,
                torch.full_like(reported_forward_speed, -torch.inf),
            )
            finite_raw_forward_speed = torch.where(
                torch.isfinite(raw_forward_speed),
                raw_forward_speed,
                torch.full_like(raw_forward_speed, -torch.inf),
            )
            rollout_forward_speed_sum = (
                rollout_forward_speed_sum
                + torch.where(
                    torch.isfinite(reported_forward_speed),
                    reported_forward_speed,
                    torch.zeros_like(reported_forward_speed),
                ).sum()
            )
            rollout_forward_speed_max = torch.maximum(rollout_forward_speed_max, finite_forward_speed.max())
            raw_rollout_forward_speed_max = torch.maximum(
                raw_rollout_forward_speed_max,
                finite_raw_forward_speed.max(),
            )

            score_sum.scatter_add_(0, particle_ids, score)
            rollout_count.scatter_add_(0, particle_ids, torch.ones_like(score))
            forward_speed_sum.scatter_add_(0, particle_ids, reported_forward_speed)
            forward_displacement_sum.scatter_add_(0, particle_ids, forward_displacement)
            lateral_displacement_sum.scatter_add_(0, particle_ids, lateral_displacement)
            final_height_sum.scatter_add_(0, particle_ids, final_height)
            survival_time_sum.scatter_add_(0, particle_ids, survival_time)
            unphysical_primary = terminal & unphysical
            fall_primary = terminal & (~unphysical_primary) & fell
            backward_primary = terminal & (~unphysical_primary) & (~fall_primary) & backward
            fall_count.scatter_add_(0, particle_ids, fall_primary.to(dtype=torch.float32))
            unphysical_count.scatter_add_(0, particle_ids, unphysical_primary.to(dtype=torch.float32))
            backward_count.scatter_add_(0, particle_ids, backward_primary.to(dtype=torch.float32))
            terminated_count.scatter_add_(0, particle_ids, terminal.to(dtype=torch.float32))
            completed_rollouts += int(env_ids.numel())

        assign_rollouts(env_ids_all[:active_envs])

        while completed_rollouts < total_rollouts and active_env_count > 0:
            active_ids = torch.nonzero(env_active, as_tuple=False).flatten()

            self.tendon_manager.apply_jit(
                virtual_ground_height=self.forrest_params.physics.virtual_ground_height,
                dt=self.sim_dt,
            )

            rollout_t = rollout_step.to(dtype=torch.float32) * self.sim_dt
            controller_t = torch.clamp(rollout_t - float(self.cfg.startup_hold_duration), min=0.0)
            active_commanded = controller_command(
                t=controller_t,
                initial_positions=initial_joint_positions,
                zero=controller_zero,
            )
            hold_mask = (rollout_t < float(self.cfg.startup_hold_duration)).unsqueeze(1)
            commanded_positions = torch.where(hold_mask, initial_joint_positions, active_commanded)
            previous_action[active_ids] = current_action[active_ids]
            current_action[active_ids] = commanded_positions[active_ids] - initial_joint_positions[active_ids]
            self.robot.set_joint_position_target(
                commanded_positions[active_ids],
                joint_ids=self.actuated_joint_indices,
                env_ids=active_ids,
            )

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim_dt)

            rollout_step[active_ids] += 1
            just_started = env_active & (~scored_started) & (rollout_step >= score_start_step)
            if torch.any(just_started):
                start_root_pos[just_started] = self.robot.data.root_pos_w[just_started]
                start_heading[just_started] = self.robot.data.heading_w[just_started]
                scored_started[just_started] = True

            scoring_active = active_ids[scored_started[active_ids]]
            if scoring_active.numel() > 0:
                terminal_reward_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
                reward, foot0_ahead_avg, foot1_ahead_avg = self._rl_reward_terms(
                    command=command_by_env,
                    terminal=terminal_reward_mask,
                    previous_action=previous_action,
                    current_action=current_action,
                    foot0_ahead_avg=foot0_ahead_avg,
                    foot1_ahead_avg=foot1_ahead_avg,
                )
                rl_reward_integral[scoring_active] += reward[scoring_active] * self.sim_dt

            root_pos = self.robot.data.root_pos_w
            root_lin_vel = self.robot.data.root_lin_vel_w
            root_ang_vel = self.robot.data.root_ang_vel_w
            relative_height = root_pos[:, 2] - self.scene.env_origins[:, 2]
            displacement = root_pos - start_root_pos

            fell_now = relative_height < float(termination_cfg.base_too_low_height)
            if termination_cfg.terminate_on_backward_progress:
                backward_allowed_to_trip = scored_started & (rollout_step >= score_start_step + backward_grace_steps)
                backward_now = backward_allowed_to_trip & (
                    displacement[:, 0] < -float(termination_cfg.max_backward_displacement)
                )
            else:
                backward_now = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

            if termination_cfg.terminate_on_unphysical:
                finite_state = (
                    torch.isfinite(root_pos).all(dim=1)
                    & torch.isfinite(root_lin_vel).all(dim=1)
                    & torch.isfinite(root_ang_vel).all(dim=1)
                )
                unphysical_now = (
                    (~finite_state)
                    | (root_lin_vel[:, 0].abs() > float(termination_cfg.max_forward_speed))
                    | (root_lin_vel[:, 1].abs() > float(termination_cfg.max_lateral_speed))
                    | (root_lin_vel[:, 2].abs() > float(termination_cfg.max_vertical_speed))
                    | (torch.linalg.norm(root_ang_vel, dim=1) > float(termination_cfg.max_root_angular_speed))
                    | (relative_height > float(termination_cfg.max_height))
                )
                if (
                    termination_cfg.terminate_on_undesired_contact
                    and self.contact_sensor is not None
                    and len(self.undesired_contact_body_indices) > 0
                ):
                    contact_allowed_to_trip = rollout_step >= undesired_contact_grace_steps
                    contact_history = self.contact_sensor.data.net_forces_w_history
                    undesired_contact = contact_history[:, :, self.undesired_contact_body_indices, :].norm(dim=-1).amax(
                        dim=(1, 2)
                    ) > float(termination_cfg.undesired_contact_force_threshold)
                    contact_step = contact_allowed_to_trip & undesired_contact
                    undesired_contact_steps[:] = torch.where(
                        env_active & contact_step,
                        undesired_contact_steps + 1,
                        torch.zeros_like(undesired_contact_steps),
                    )
                    unphysical_now = unphysical_now | (undesired_contact_steps >= undesired_contact_consecutive_steps)
                if termination_cfg.terminate_on_joint_vibration and len(self.joint_vibration_joint_indices) > 0:
                    vibration_allowed_to_trip = rollout_step >= joint_vibration_grace_steps
                    joint_vel = self.robot.data.joint_vel[:, self.joint_vibration_joint_indices]
                    joint_acc = self.robot.data.joint_acc[:, self.joint_vibration_joint_indices]
                    finite_joints = torch.isfinite(joint_vel).all(dim=1) & torch.isfinite(joint_acc).all(dim=1)
                    vibration_step = vibration_allowed_to_trip & (
                        (~finite_joints)
                        | (joint_vel.abs().amax(dim=1) > float(termination_cfg.max_joint_vibration_velocity))
                        | (joint_acc.abs().amax(dim=1) > float(termination_cfg.max_joint_vibration_acceleration))
                    )
                    joint_vibration_steps[:] = torch.where(
                        env_active & vibration_step,
                        joint_vibration_steps + 1,
                        torch.zeros_like(joint_vibration_steps),
                    )
                    unphysical_now = unphysical_now | (joint_vibration_steps >= joint_vibration_consecutive_steps)
            else:
                unphysical_now = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

            reached_duration = rollout_step >= num_steps
            done = env_active & (fell_now | unphysical_now | backward_now | reached_duration)
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if done_ids.numel() == 0:
                continue

            finalize_rollouts(
                done_ids,
                fell=fell_now[done_ids],
                unphysical=unphysical_now[done_ids],
                backward=backward_now[done_ids],
                terminal=(fell_now | unphysical_now | backward_now)[done_ids],
            )
            env_active[done_ids] = False
            active_env_count -= int(done_ids.numel())
            assign_rollouts(done_ids)

        count = torch.clamp(rollout_count, min=1.0)
        scores = torch.where(rollout_count > 0, score_sum / count, torch.full_like(score_sum, -torch.inf))
        forward_speed = forward_speed_sum / count
        forward_displacement = forward_displacement_sum / count
        lateral_displacement = lateral_displacement_sum / count
        final_height = final_height_sum / count
        mean_survival_time = float((survival_time_sum.sum() / torch.clamp(rollout_count.sum(), min=1.0)).detach().cpu())

        completed = max(1, completed_rollouts)
        return EvaluationResult(
            scores=scores.detach(),
            forward_speed=forward_speed.detach(),
            forward_displacement=forward_displacement.detach(),
            lateral_displacement=lateral_displacement.detach(),
            final_height=final_height.detach(),
            fell=(fall_count > 0).detach(),
            unphysical=(unphysical_count > 0).detach(),
            backward=(backward_count > 0).detach(),
            terminated=(terminated_count > 0).detach(),
            completed_rollouts=completed_rollouts,
            fall_percent=float((fall_count.sum() / completed * 100.0).detach().cpu()),
            unphysical_percent=float((unphysical_count.sum() / completed * 100.0).detach().cpu()),
            backward_percent=float((backward_count.sum() / completed * 100.0).detach().cpu()),
            terminated_percent=float((terminated_count.sum() / completed * 100.0).detach().cpu()),
            mean_survival_time=mean_survival_time,
            mean_rollout_forward_speed=float((rollout_forward_speed_sum / completed).detach().cpu()),
            max_rollout_forward_speed=float(rollout_forward_speed_max.detach().cpu()),
            raw_max_rollout_forward_speed=float(raw_rollout_forward_speed_max.detach().cpu()),
        )
