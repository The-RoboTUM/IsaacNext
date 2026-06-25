# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_tracking_module():
    module_path = Path(__file__).resolve().parents[1] / "isaaclab_rl" / "rsl_rl" / "experiment_tracking.py"
    spec = importlib.util.spec_from_file_location("experiment_tracking_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_profiling_module():
    module_path = Path(__file__).resolve().parents[1] / "isaaclab_rl" / "rsl_rl" / "profiling.py"
    spec = importlib.util.spec_from_file_location("profiling_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ppo_loss_mapping_uses_stable_names():
    tracking = _load_tracking_module()

    scalars = tracking._map_ppo_losses({"value": 1.0, "surrogate": 2.0, "entropy": torch.tensor(0.5)})

    assert scalars == {
        "ppo/value_loss": 1.0,
        "ppo/policy_loss": 2.0,
        "ppo/entropy": 0.5,
    }


def test_checkpoint_infos_restore_rng_and_env_steps():
    tracking = _load_tracking_module()
    before = torch.rand(3)

    infos = tracking.collect_checkpoint_infos(runner=object(), env_steps=123)
    after = torch.rand(3)
    restored_steps = tracking.restore_checkpoint_infos(infos)
    replay = torch.rand(3)

    assert restored_steps == 123
    assert not torch.equal(before, after)
    torch.testing.assert_close(after, replay)


def test_noop_logger_accepts_tracking_calls(tmp_path):
    tracking = _load_tracking_module()
    logger = tracking.create_experiment_logger(False)
    path = tmp_path / "metadata.json"
    path.write_text("{}", encoding="utf-8")

    logger.log_scalars({"train/fps": 10}, step=1)
    logger.log_file(path)
    logger.log_artifact(path, name="checkpoint", artifact_type="checkpoint", aliases=["latest"])
    logger.close()


def test_tracking_logger_uses_iteration_step_for_wandb_compatibility():
    tracking = _load_tracking_module()
    tracker = _FakeTracker(enabled=True)
    wrapped = _FakeRslRlLogger()
    proxy = tracking.TrackingLoggerProxy(wrapped=wrapped, tracker=tracker, env=None, log_dir="logs/test")
    proxy.env_steps = 294912

    proxy.log(
        it=3,
        start_it=0,
        total_it=10,
        collect_time=1.0,
        learn_time=1.0,
        loss_dict={"value": 1.0, "surrogate": 2.0, "entropy": 0.5},
        learning_rate=0.001,
        action_std=torch.ones(1),
        rnd_weight=None,
    )

    assert tracker.logged_step == 3
    assert tracker.logged_scalars["train/env_steps"] == 294912


def test_tracking_logger_interval_skips_wrapped_log_but_advances_state():
    tracking = _load_tracking_module()
    tracker = _FakeTracker(enabled=False)
    wrapped = _FakeRslRlLogger()
    proxy = tracking.TrackingLoggerProxy(
        wrapped=wrapped,
        tracker=tracker,
        env=None,
        log_dir="logs/test",
        options=tracking.TrackingOptions(log_interval=10),
    )
    wrapped.ep_extras.append({"Episode_Reward/alive": 1.0})

    proxy.log(
        it=3,
        start_it=0,
        total_it=10,
        collect_time=1.0,
        learn_time=2.0,
        loss_dict={},
        learning_rate=0.001,
        action_std=torch.ones(1),
        rnd_weight=None,
    )

    assert wrapped.log_calls == 0
    assert wrapped.tot_timesteps == 24 * 4096
    assert wrapped.tot_time == 3.0
    assert wrapped.ep_extras == []

    proxy.log(
        it=10,
        start_it=0,
        total_it=10,
        collect_time=1.0,
        learn_time=2.0,
        loss_dict={},
        learning_rate=0.001,
        action_std=torch.ones(1),
        rnd_weight=None,
    )

    assert wrapped.log_calls == 1


def test_episode_reward_and_termination_are_not_mirrored():
    tracking = _load_tracking_module()
    proxy = tracking.TrackingLoggerProxy(
        wrapped=_FakeRslRlLogger(),
        tracker=_FakeTracker(enabled=True),
        env=None,
        log_dir="logs/test",
    )
    proxy._episode_scalars["Episode_Reward/alive"].append(1.0)
    proxy._episode_scalars["Episode_Termination/base_contact"].append(1.0)
    proxy._episode_scalars["Metrics/base_velocity/error_vel_xy"].append(0.5)

    scalars = proxy._map_episode_scalars()

    assert "reward/weighted/alive" not in scalars
    assert "termination/base_contact" not in scalars
    assert scalars["Metrics/base_velocity/error_vel_xy"] == 0.5


def test_noop_tracking_uses_save_time_state_without_wrapping_logger(tmp_path):
    tracking = _load_tracking_module()
    runner = _FakeRunner(tmp_path)
    original_logger = runner.logger

    state = tracking.install_tracking_hooks(
        runner=runner,
        tracker=tracking.NoOpExperimentLogger(),
        log_dir=tmp_path,
        options=tracking.TrackingOptions(extra_metrics_interval=1),
    )
    runner.save(str(tmp_path / "model_2.pt"))

    assert runner.logger is original_logger
    assert type(state).__name__ == "CheckpointTrackingState"
    assert runner.saved_infos["tracking"]["env_steps"] == 2 * 24 * 4
    assert (tmp_path / "latest.pt").exists()


def test_profiler_wraps_methods_and_writes_summary(tmp_path):
    profiling = _load_profiling_module()
    runner = _FakeRunner(tmp_path)

    profiler = profiling.install_training_profiler(runner, log_dir=tmp_path, enabled=True)
    runner.alg.act("obs")
    runner.env.step("actions")
    runner.alg.update()
    summary_path = profiler.write_summary()

    assert summary_path is not None
    assert summary_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "rollout/policy_act" in summary
    assert "rollout/env_step" in summary
    assert "train/ppo_update" in summary


class _FakeTracker:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.logged_scalars = None
        self.logged_step = None

    def log_scalars(self, scalars, step=None):
        self.logged_scalars = scalars
        self.logged_step = step

    def close(self):
        return


class _FakeRslRlLogger:
    cfg = {"num_steps_per_env": 24}
    num_envs = 4096
    gpu_world_size = 1
    writer = object()

    def __init__(self):
        self.log_calls = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.ep_extras = []

    def log(self, *args, **kwargs):
        self.log_calls += 1
        return

    def process_env_step(self, *args, **kwargs):
        return


class _FakeRunner:
    def __init__(self, log_dir):
        self.logger = _FakeRslRlLogger()
        self.alg = _FakeAlg()
        self.env = _FakeEnv()
        self.cfg = {"num_steps_per_env": 24}
        self.current_learning_iteration = 2
        self.log_dir = log_dir
        self.saved_infos = None

    def save(self, path, infos=None):
        Path(path).write_bytes(b"checkpoint")
        self.saved_infos = infos


class _FakeAlg:
    learning_rate = 0.001

    def act(self, obs):
        return obs

    def process_env_step(self, *args, **kwargs):
        return

    def compute_returns(self, obs):
        return obs

    def update(self):
        return {"value": 1.0}


class _FakeEnv:
    num_envs = 4
    device = "cpu"

    def step(self, actions):
        return actions
