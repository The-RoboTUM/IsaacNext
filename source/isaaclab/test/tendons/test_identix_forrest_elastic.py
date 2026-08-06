# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from isaaclab.tendons.models.identix.forrest_elastic import (
    FULL_ROBOT_NUM_DOFS,
    REAL_LEG_JOINTS_LEFT,
    REAL_LEG_JOINTS_RIGHT,
    IdentixForrestElasticTendonModel,
    _configure_jax_logging,
)
from isaaclab.tendons.parameter_loader import load_forrest_parameter_config


class FakeWrenchComposer:
    def __init__(self):
        self.calls = []

    def set_forces_and_torques(self, *, forces, torques, body_ids):
        self.calls.append((forces.clone(), torques.clone(), list(body_ids)))


class FakeRobot:
    def __init__(self):
        self.device = "cpu"
        self.num_instances = 2
        self.joint_names = [*REAL_LEG_JOINTS_RIGHT, *REAL_LEG_JOINTS_LEFT]
        joint_pos = torch.arange(2 * len(self.joint_names), dtype=torch.float32).reshape(2, len(self.joint_names))
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            joint_pos=joint_pos,
        )
        self.permanent_wrench_composer = FakeWrenchComposer()
        self.effort_targets = []

    def find_joints(self, joint_names, preserve_order=True):
        del preserve_order
        indices = [self.joint_names.index(name) for name in joint_names if name in self.joint_names]
        found = [self.joint_names[index] for index in indices]
        return indices, found

    def find_bodies(self, body_names, preserve_order=True):
        del preserve_order
        return list(range(len(body_names))), list(body_names)

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        self.effort_targets.append((target.clone(), list(joint_ids), env_ids))


class FakeDeployment:
    def elastic_force_batch(self, q_full):
        force = q_full * 0.0
        force[:, [11, 12, 15, 16, 17]] = 1.0
        force[:, [23, 24, 27, 28, 29]] = 2.0
        return force


def test_default_forrest_config_selects_identix_tendon_model():
    params = load_forrest_parameter_config()

    assert params.training.actions.tendon_model_type == "identix_elastic"
    assert params.training.actions.tendon_identix_bundle_dir
    assert params.training.actions.tendon_identix_repo_path


def test_identix_jax_logging_defaults_to_warning(monkeypatch):
    monkeypatch.delenv("JAX_LOG_COMPILES", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("ISAACLAB_IDENTIX_JAX_LOG_LEVEL", raising=False)
    logging.getLogger("jax").setLevel(logging.NOTSET)
    logging.getLogger("jax._src").setLevel(logging.NOTSET)

    _configure_jax_logging()

    assert logging.getLogger("jax").level == logging.WARNING
    assert logging.getLogger("jax._src").level == logging.WARNING


def test_identix_forrest_q_packing_uses_full_robot_recording_order(tmp_path):
    robot = FakeRobot()
    model = IdentixForrestElasticTendonModel(
        robot,
        bundle_dir=tmp_path,
        identix_repo_path=tmp_path,
        deployment=object(),
    )

    q_full = model.full_q_from_robot(robot)

    assert q_full.shape == (robot.num_instances, FULL_ROBOT_NUM_DOFS)
    torch.testing.assert_close(q_full[:, 0:3], robot.data.root_pos_w)
    torch.testing.assert_close(q_full[:, 3:6], torch.zeros((robot.num_instances, 3)))
    torch.testing.assert_close(
        q_full[:, 6:18],
        robot.data.joint_pos[:, [robot.joint_names.index(name) for name in REAL_LEG_JOINTS_LEFT]],
    )
    torch.testing.assert_close(
        q_full[:, 18:30],
        robot.data.joint_pos[:, [robot.joint_names.index(name) for name in REAL_LEG_JOINTS_RIGHT]],
    )


def test_identix_force_extracts_tendon_chain_joint_torques(tmp_path):
    robot = FakeRobot()
    model = IdentixForrestElasticTendonModel(
        robot,
        bundle_dir=tmp_path,
        identix_repo_path=tmp_path,
        force_scale=2.0,
        force_sign=-1.0,
        deployment=object(),
    )
    force_full = torch.zeros((2, FULL_ROBOT_NUM_DOFS), dtype=torch.float32)
    force_full[:, [11, 12, 15, 16, 17]] = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
    force_full[:, [23, 24, 27, 28, 29]] = torch.tensor([[11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0]])

    left, right = model.tendon_torques_from_full_force(force_full)

    torch.testing.assert_close(left, -2.0 * force_full[:, [11, 12, 15, 16, 17]])
    torch.testing.assert_close(right, -2.0 * force_full[:, [23, 24, 27, 28, 29]])


def test_identix_joint_torques_uses_configured_deployment(tmp_path):
    robot = FakeRobot()
    model = IdentixForrestElasticTendonModel(
        robot,
        bundle_dir=tmp_path,
        identix_repo_path=tmp_path,
        transfer="numpy",
        deployment=FakeDeployment(),
    )

    left, right = model.joint_torques(robot)

    torch.testing.assert_close(left, torch.ones((robot.num_instances, 5)))
    torch.testing.assert_close(right, 2.0 * torch.ones((robot.num_instances, 5)))


def test_identix_joint_effort_apply_mode_bypasses_link_torque_mapper(tmp_path):
    manager_module = pytest.importorskip("isaaclab.tendons.manager", exc_type=ImportError)
    robot = FakeRobot()
    model = IdentixForrestElasticTendonModel(
        robot,
        bundle_dir=tmp_path,
        identix_repo_path=tmp_path,
        transfer="numpy",
        deployment=FakeDeployment(),
    )
    manager = manager_module.TendonManager(robot, model=model, apply_mode="joint_effort")

    left, right = manager.apply_jit(dt=0.0024)

    assert len(robot.effort_targets) == 2
    torch.testing.assert_close(robot.effort_targets[0][0], left)
    torch.testing.assert_close(robot.effort_targets[1][0], right)
    assert robot.permanent_wrench_composer.calls
    torch.testing.assert_close(robot.permanent_wrench_composer.calls[-1][1], torch.zeros((robot.num_instances, 12, 3)))
    assert manager.cached_tendon_link_torques is None
    assert manager.cached_tendon_body_ids is None
    assert manager.cached_tendon_joint_torques is not None
