# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class TimingRecord:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    def to_dict(self) -> dict[str, float | int]:
        mean_s = self.total_s / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_s": self.total_s,
            "mean_s": mean_s,
            "max_s": self.max_s,
        }


@dataclass
class TrainingProfiler:
    """Small wall-clock profiler for RSL-RL training phases."""

    log_dir: str | Path
    enabled: bool = False
    records: dict[str, TimingRecord] = field(default_factory=dict)

    @contextmanager
    def scope(self, name: str):
        if not self.enabled:
            yield
            return
        self._synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            elapsed_s = time.perf_counter() - start
            self.record_elapsed(name, elapsed_s)

    def record_elapsed(self, name: str, elapsed_s: float) -> None:
        if not self.enabled:
            return
        self.records.setdefault(name, TimingRecord()).add(elapsed_s)

    def wrap_method(self, owner: Any, method_name: str, label: str) -> None:
        if not self.enabled or not hasattr(owner, method_name):
            return
        original = getattr(owner, method_name)

        def wrapped(*args, **kwargs):
            with self.scope(label):
                return original(*args, **kwargs)

        setattr(owner, method_name, wrapped)

    def write_summary(self) -> Path | None:
        if not self.enabled:
            return None
        output_dir = Path(self.log_dir) / "profile"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "profile_summary.json"
        payload = {
            "records": {name: record.to_dict() for name, record in sorted(self.records.items())},
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return output_path

    def _synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def install_training_profiler(runner: Any, *, log_dir: str | Path, enabled: bool) -> TrainingProfiler:
    profiler = TrainingProfiler(log_dir=log_dir, enabled=enabled)
    if not enabled:
        return profiler

    profiler.wrap_method(runner.alg, "act", "rollout/policy_act")
    profiler.wrap_method(runner.env, "step", "rollout/env_step")
    profiler.wrap_method(runner.alg, "process_env_step", "rollout/process_env_step")
    profiler.wrap_method(runner.logger, "process_env_step", "logging/process_env_step")
    profiler.wrap_method(runner.alg, "compute_returns", "train/compute_returns")
    profiler.wrap_method(runner.alg, "update", "train/ppo_update")
    profiler.wrap_method(runner.logger, "log", "logging/log")
    profiler.wrap_method(runner, "save", "checkpoint/save")
    env = getattr(runner.env, "unwrapped", runner.env)
    setattr(env, "_external_profiler", profiler)
    return profiler
