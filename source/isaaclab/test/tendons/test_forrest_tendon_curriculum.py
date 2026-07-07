# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused checks for Forrest tendon and command-curriculum utilities."""

import math

import torch

from isaaclab.curriculums.command_bins import (
    CommandBinCurriculumParameters,
    CommandBinCurriculumState,
    command_tracking_reward_success,
    command_tracking_success,
)
from isaaclab.pso.config import PsoConfig
from isaaclab.pso.kernels import cpg_oscillator_command_kernel
from isaaclab.pso.parameters import ParameterSpace
from isaaclab.tendons.controllers.base import DOF_ORDER, DOF_SIGN
from isaaclab.tendons.controllers.cpg import BirdBotCPGLeg, CPGParams, HopfCPGLeg, HopfCPGParams
from isaaclab.tendons.controllers.sinusoidal import SinusoidalLegController, SinusoidalParams
from isaaclab.tendons.models.analytic.analytic_energy_model import AnalyticTendonEnergyModel
from isaaclab.tendons.models.analytic.tendon_data import TendonData
from isaaclab.tendons.parameter_loader import RunSinusoidalControllerParameters, load_forrest_parameter_config
from isaaclab.tendons.runner import (
    ActuatedDofSpec,
    cpg_command_batch,
    cpg_oscillator_command_batch,
    sinusoidal_command_batch,
)
from isaaclab.tendons.torque_mapper import TendonTorqueMapper


def _actuated_specs() -> list[ActuatedDofSpec]:
    return [
        ActuatedDofSpec(side=side, dof=dof, joint_expr=f"{side}_{dof}", sign=DOF_SIGN[dof])
        for side in ("left", "right")
        for dof in DOF_ORDER
    ]


def test_sinusoidal_default_right_phase_is_radians():
    params = RunSinusoidalControllerParameters()

    assert params.right_phase_rad["hip_flexion"] == math.pi
    assert params.right_phase_rad["knee_flexion"] == math.pi


def test_pso_exported_forrest_yaml_loads_from_external_directory(tmp_path):
    cfg = PsoConfig.from_yaml("configs/pso.yaml")
    space = ParameterSpace(cfg.parameters, device="cpu")
    physical = space.denormalize(space.initial_normalized())
    output_path = tmp_path / "best.yaml"

    space.export_forrest_yaml(output_path, physical, includes=["configs/forrest/default"])
    loaded = load_forrest_parameter_config(output_path)
    initial_by_name = {param.name: param.initial for param in cfg.parameters}

    assert loaded.run.controller == "cpg"
    assert math.isclose(
        loaded.tendons.baseline.lengths["gst_spring_rest"],
        initial_by_name["tendons.baseline.lengths.gst_spring_rest"],
        rel_tol=1e-6,
    )
    assert math.isclose(loaded.run.cpg.f_hz, initial_by_name["run.cpg.f_hz"], rel_tol=1e-6)


def test_tendon_data_and_jit_energy_support_cpu():
    params = load_forrest_parameter_config()
    tendon_data = TendonData(
        3,
        params.to_tendon_randomization_ranges(),
        tc=params.to_tendon_constants(device="cpu"),
        device="cpu",
    )
    model = AnalyticTendonEnergyModel(tendon_data)
    joint_angles = torch.tensor(
        [
            [-6.0, -3.0, 0.0, 3.0, 6.0],
            [6.0, 3.0, 0.0, -3.0, -6.0],
            [1.5, -2.5, 3.5, -4.5, 5.5],
            [-1.5, 2.5, -3.5, 4.5, -5.5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
        ],
        dtype=torch.float32,
        device="cpu",
        requires_grad=True,
    )

    deltas = model.delta_lengths_jit(joint_angles)
    energy = model.energy_from_delta_lengths_jit(deltas)

    assert energy.device.type == "cpu"
    assert torch.isfinite(energy)
    for delta in deltas:
        assert delta.device.type == "cpu"
        assert torch.isfinite(delta).all()


def test_torque_mapper_jit_matches_eager_mapping():
    mapper = TendonTorqueMapper(device="cpu")
    left = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [-2.0, 0.5, 1.5, -3.0, 0.25],
        ],
        dtype=torch.float32,
    )
    right = -0.5 * left

    eager = mapper.joint_to_link_torques(left, right, batch_size=left.shape[0])
    scripted = mapper.joint_to_link_torques_jit(left, right, batch_size=left.shape[0]).clone()

    torch.testing.assert_close(scripted, eager)


def test_vectorized_sinusoidal_command_matches_scalar_controller():
    specs = _actuated_specs()
    initial_joint_positions = torch.zeros((1, len(specs)), dtype=torch.float32)
    controller_zero = torch.zeros_like(initial_joint_positions)
    params = RunSinusoidalControllerParameters()
    t = 0.37

    left = SinusoidalLegController(
        SinusoidalParams(
            f_hz=params.f_hz,
            phi0=params.left_phi0_rad,
            amplitude_deg=params.amplitude_deg,
            offset_deg=params.offset_deg,
            phase_rad=params.left_phase_rad,
        )
    )
    right = SinusoidalLegController(
        SinusoidalParams(
            f_hz=params.f_hz,
            phi0=params.right_phi0_rad,
            amplitude_deg=params.amplitude_deg,
            offset_deg=params.offset_deg,
            phase_rad=params.right_phase_rad,
        )
    )
    params_by_env = {
        "run.sinusoidal.f_hz": torch.tensor([params.f_hz]),
        "run.sinusoidal.left_phi0_rad": torch.tensor([params.left_phi0_rad]),
        "run.sinusoidal.right_phi0_rad": torch.tensor([params.right_phi0_rad]),
    }
    for dof in DOF_ORDER:
        params_by_env[f"run.sinusoidal.amplitude_deg.{dof}"] = torch.tensor([params.amplitude_deg[dof]])
        params_by_env[f"run.sinusoidal.offset_deg.{dof}"] = torch.tensor([params.offset_deg[dof]])
        params_by_env[f"run.sinusoidal.left_phase_rad.{dof}"] = torch.tensor([params.left_phase_rad.get(dof, 0.0)])
        params_by_env[f"run.sinusoidal.right_phase_rad.{dof}"] = torch.tensor([params.right_phase_rad.get(dof, 0.0)])

    batch = sinusoidal_command_batch(
        t=t,
        params_by_env=params_by_env,
        actuated_dof_specs=specs,
        initial_joint_positions=initial_joint_positions,
        controller_zero=controller_zero,
    )
    expected = torch.tensor(
        [[spec.sign * (left if spec.side == "left" else right).joint(spec.dof, t)[0] for spec in specs]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(batch, expected, rtol=1e-6, atol=1e-6)


def test_vectorized_basic_cpg_command_matches_scalar_controller():
    specs = _actuated_specs()
    initial_joint_positions = torch.zeros((1, len(specs)), dtype=torch.float32)
    controller_zero = torch.zeros_like(initial_joint_positions)
    t = 0.37
    common = {
        "f_hz": 1.1,
        "D": 0.62,
        "A_h_deg": 27.0,
        "O_h_deg": 5.0,
        "A_k_deg": 95.0,
        "S_f": 0.04,
        "S_e": 0.08,
    }
    left = BirdBotCPGLeg(CPGParams(phi0=-0.35, **common), include_knee=True)
    right = BirdBotCPGLeg(CPGParams(phi0=2.1, **common), include_knee=True)
    params_by_env = {
        "run.cpg.f_hz": torch.tensor([common["f_hz"]]),
        "run.cpg.duty_factor": torch.tensor([common["D"]]),
        "run.cpg.hip_amplitude_deg": torch.tensor([common["A_h_deg"]]),
        "run.cpg.hip_offset_deg": torch.tensor([common["O_h_deg"]]),
        "run.cpg.knee_amplitude_deg": torch.tensor([common["A_k_deg"]]),
        "run.cpg.swing_start_offset": torch.tensor([common["S_f"]]),
        "run.cpg.swing_end_offset": torch.tensor([common["S_e"]]),
        "run.cpg.combined_phase_offset_rad": torch.tensor([0.0]),
        "run.cpg.left_phase_offset_rad": torch.tensor([-0.35]),
        "run.cpg.right_phase_offset_rad": torch.tensor([2.1]),
    }

    batch = cpg_command_batch(
        t=t,
        params_by_env=params_by_env,
        actuated_dof_specs=specs,
        initial_joint_positions=initial_joint_positions,
        controller_zero=controller_zero,
    )
    expected = torch.tensor(
        [[spec.sign * (left if spec.side == "left" else right).joint(spec.dof, t)[0] for spec in specs]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(batch, expected, rtol=1e-6, atol=1e-6)


def test_vectorized_cpg_oscillator_command_matches_scalar_controller():
    specs = _actuated_specs()
    initial_joint_positions = torch.zeros((1, len(specs)), dtype=torch.float32)
    controller_zero = torch.zeros_like(initial_joint_positions)
    left_params = HopfCPGParams(
        f_hz=0.93,
        duty_factor=0.58,
        phi0=-0.4,
        hip_flexion_amplitude_deg=19.0,
        hip_flexion_offset_deg=6.5,
        knee_flexion_amplitude_deg=41.0,
        knee_flexion_phase_rad=1.1,
        knee_swing_power=1.7,
        hip_roll_amplitude_deg=3.0,
        hip_roll_phase_rad=0.2,
        hip_yaw_amplitude_deg=4.0,
        hip_yaw_phase_rad=-0.1,
    )
    right_params = HopfCPGParams(**{**left_params.__dict__, "phi0": 2.2})
    left = HopfCPGLeg(left_params)
    right = HopfCPGLeg(right_params)
    t = 0.41
    params_by_env = {
        "run.cpg_oscillator.f_hz": torch.tensor([left_params.f_hz]),
        "run.cpg_oscillator.duty_factor": torch.tensor([left_params.duty_factor]),
        "run.cpg_oscillator.left_phase_rad": torch.tensor([left_params.phi0]),
        "run.cpg_oscillator.right_phase_rad": torch.tensor([right_params.phi0]),
        "run.cpg_oscillator.hip_flexion_amplitude_deg": torch.tensor([left_params.hip_flexion_amplitude_deg]),
        "run.cpg_oscillator.hip_flexion_offset_deg": torch.tensor([left_params.hip_flexion_offset_deg]),
        "run.cpg_oscillator.hip_flexion_phase_rad": torch.tensor([left_params.hip_flexion_phase_rad]),
        "run.cpg_oscillator.knee_flexion_amplitude_deg": torch.tensor([left_params.knee_flexion_amplitude_deg]),
        "run.cpg_oscillator.knee_flexion_offset_deg": torch.tensor([left_params.knee_flexion_offset_deg]),
        "run.cpg_oscillator.knee_flexion_phase_rad": torch.tensor([left_params.knee_flexion_phase_rad]),
        "run.cpg_oscillator.knee_swing_power": torch.tensor([left_params.knee_swing_power]),
        "run.cpg_oscillator.hip_roll_amplitude_deg": torch.tensor([left_params.hip_roll_amplitude_deg]),
        "run.cpg_oscillator.hip_roll_offset_deg": torch.tensor([left_params.hip_roll_offset_deg]),
        "run.cpg_oscillator.hip_roll_phase_rad": torch.tensor([left_params.hip_roll_phase_rad]),
        "run.cpg_oscillator.hip_yaw_amplitude_deg": torch.tensor([left_params.hip_yaw_amplitude_deg]),
        "run.cpg_oscillator.hip_yaw_offset_deg": torch.tensor([left_params.hip_yaw_offset_deg]),
        "run.cpg_oscillator.hip_yaw_phase_rad": torch.tensor([left_params.hip_yaw_phase_rad]),
    }

    batch = cpg_oscillator_command_batch(
        t=t,
        params_by_env=params_by_env,
        actuated_dof_specs=specs,
        initial_joint_positions=initial_joint_positions,
        controller_zero=controller_zero,
    )
    expected = torch.tensor(
        [[spec.sign * (left if spec.side == "left" else right).joint(spec.dof, t)[0] for spec in specs]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(batch, expected, rtol=1e-6, atol=1e-6)


def test_scripted_cpg_oscillator_kernel_matches_python_batch():
    specs = _actuated_specs()
    initial_joint_positions = torch.full((2, len(specs)), 0.2, dtype=torch.float32)
    controller_zero = torch.full_like(initial_joint_positions, -0.1)
    t = torch.tensor([0.15, 0.57], dtype=torch.float32)
    params_by_env = {
        "run.cpg_oscillator.f_hz": torch.tensor([0.8, 1.2]),
        "run.cpg_oscillator.duty_factor": torch.tensor([0.58, 0.64]),
        "run.cpg_oscillator.left_phase_rad": torch.tensor([0.0, 0.2]),
        "run.cpg_oscillator.right_phase_rad": torch.tensor([math.pi, 2.4]),
        "run.cpg_oscillator.hip_flexion_amplitude_deg": torch.tensor([24.0, 31.0]),
        "run.cpg_oscillator.hip_flexion_offset_deg": torch.tensor([8.0, -4.0]),
        "run.cpg_oscillator.hip_flexion_phase_rad": torch.tensor([0.0, 0.3]),
        "run.cpg_oscillator.knee_flexion_amplitude_deg": torch.tensor([34.0, 42.0]),
        "run.cpg_oscillator.knee_flexion_offset_deg": torch.tensor([0.0, 5.0]),
        "run.cpg_oscillator.knee_flexion_phase_rad": torch.tensor([math.pi / 2.0, 1.2]),
        "run.cpg_oscillator.knee_swing_power": torch.tensor([1.5, 2.1]),
        "run.cpg_oscillator.hip_roll_amplitude_deg": torch.tensor([3.0, 7.0]),
        "run.cpg_oscillator.hip_roll_offset_deg": torch.tensor([0.0, -1.0]),
        "run.cpg_oscillator.hip_roll_phase_rad": torch.tensor([0.1, 0.4]),
        "run.cpg_oscillator.hip_yaw_amplitude_deg": torch.tensor([4.0, 8.0]),
        "run.cpg_oscillator.hip_yaw_offset_deg": torch.tensor([0.0, 2.0]),
        "run.cpg_oscillator.hip_yaw_phase_rad": torch.tensor([-0.2, 0.6]),
    }
    joint_side_ids = torch.tensor([0 if spec.side == "left" else 1 for spec in specs], dtype=torch.long)
    dof_ids = {"hip_roll": 0, "hip_yaw": 1, "hip_flexion": 2, "knee_flexion": 3}
    joint_dof_ids = torch.tensor([dof_ids[spec.dof] for spec in specs], dtype=torch.long)
    joint_signs = torch.tensor([spec.sign for spec in specs], dtype=torch.float32)

    python_batch = cpg_oscillator_command_batch(
        t=t,
        params_by_env=params_by_env,
        actuated_dof_specs=specs,
        initial_joint_positions=initial_joint_positions,
        controller_zero=controller_zero,
    )
    scripted = cpg_oscillator_command_kernel(
        t,
        initial_joint_positions,
        controller_zero,
        joint_side_ids,
        joint_dof_ids,
        joint_signs,
        params_by_env["run.cpg_oscillator.f_hz"],
        params_by_env["run.cpg_oscillator.duty_factor"],
        params_by_env["run.cpg_oscillator.left_phase_rad"],
        params_by_env["run.cpg_oscillator.right_phase_rad"],
        torch.deg2rad(params_by_env["run.cpg_oscillator.hip_flexion_amplitude_deg"]),
        torch.deg2rad(params_by_env["run.cpg_oscillator.hip_flexion_offset_deg"]),
        params_by_env["run.cpg_oscillator.hip_flexion_phase_rad"],
        torch.deg2rad(params_by_env["run.cpg_oscillator.knee_flexion_amplitude_deg"]),
        torch.deg2rad(params_by_env["run.cpg_oscillator.knee_flexion_offset_deg"]),
        params_by_env["run.cpg_oscillator.knee_flexion_phase_rad"],
        params_by_env["run.cpg_oscillator.knee_swing_power"],
        torch.deg2rad(params_by_env["run.cpg_oscillator.hip_roll_amplitude_deg"]),
        torch.deg2rad(params_by_env["run.cpg_oscillator.hip_roll_offset_deg"]),
        params_by_env["run.cpg_oscillator.hip_roll_phase_rad"],
        torch.deg2rad(params_by_env["run.cpg_oscillator.hip_yaw_amplitude_deg"]),
        torch.deg2rad(params_by_env["run.cpg_oscillator.hip_yaw_offset_deg"]),
        params_by_env["run.cpg_oscillator.hip_yaw_phase_rad"],
    )

    torch.testing.assert_close(scripted, python_batch, rtol=1e-6, atol=1e-6)


def test_command_tracking_success_checks_yaw_and_termination():
    command = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    displacement_xy = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    heading_delta = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    duration = torch.ones(4, dtype=torch.float32)
    terminated = torch.tensor([False, False, False, True])

    success = command_tracking_success(
        command,
        displacement_xy,
        heading_delta,
        duration,
        terminated,
        0.5,
        0.05,
        0.05,
        True,
        True,
    )

    assert success.tolist() == [True, True, False, False]

    x_only_success = command_tracking_success(
        command,
        displacement_xy,
        heading_delta,
        duration,
        terminated,
        0.5,
        0.05,
        0.05,
        True,
        False,
    )

    assert x_only_success.tolist() == [True, True, True, False]


def test_command_tracking_reward_success_uses_reward_threshold_and_survival():
    tracking_reward = torch.tensor([0.85, 0.79, 0.90, 0.95], dtype=torch.float32)
    duration = torch.tensor([6.0, 6.0, 5.9, 6.0], dtype=torch.float32)
    terminated = torch.tensor([False, False, False, True])

    success = command_tracking_reward_success(
        tracking_reward,
        duration,
        terminated,
        6.0,
        0.8,
    )

    assert success.tolist() == [True, False, False, False]


def test_command_curriculum_samples_lookahead_window_without_poisoning_future_bins():
    params = CommandBinCurriculumParameters(
        include_stand_bin=True,
        lin_vel_x_min=0.0,
        lin_vel_x_max=1.0,
        lin_vel_x_bin_width=0.1,
        initial_unlocked_bin=2,
        sample_lookahead_lin_vel_x=0.5,
    )
    state = CommandBinCurriculumState(params, device="cpu")

    commands, bin_ids = state.sample(512)

    assert int(bin_ids.min()) >= 2
    assert int(bin_ids.max()) <= 7
    assert torch.all(commands[:, 0] >= 0.1)
    assert torch.all(commands[:, 0] <= 0.7)

    state.update(torch.tensor([2, 3, 7], dtype=torch.long), torch.tensor([True, True, True]))

    assert state.attempts[2].item() == 1.0
    assert state.successes[2].item() == 1.0
    assert state.attempts[3].item() == 0.0
    assert state.attempts[7].item() == 0.0
