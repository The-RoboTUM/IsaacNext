# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch implementation of bounded global-best PSO."""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.pso.config import SwarmConfig
from isaaclab.pso.kernels import pso_integrate_step, ring_best_positions


class TorchPso:
    """Bounded PSO operating in normalized ``[0, 1]`` coordinates."""

    def __init__(
        self,
        cfg: SwarmConfig,
        *,
        dim: int,
        device: str | torch.device,
        initial_position: torch.Tensor | None = None,
    ):
        self.cfg = cfg
        self.dim = dim
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(cfg.seed))
        self.iterations_since_global_improvement = 0

        self.positions = self._initial_positions()
        if initial_position is not None:
            self.positions[0] = initial_position.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)

        vel_scale = float(cfg.velocity_clamp)
        self.velocities = (
            2.0 * torch.rand(self.positions.shape, generator=self.generator, device=self.device) - 1.0
        ) * vel_scale
        self.personal_best_positions = self.positions.clone()
        self.personal_best_scores = torch.full((cfg.num_particles,), -torch.inf, device=self.device)
        self.global_best_position = self.positions[0].clone()
        self.global_best_score = torch.tensor(-torch.inf, device=self.device)
        self.iteration = 0
        self.async_updates = 0

    def _initial_positions(self) -> torch.Tensor:
        initialization = str(self.cfg.initialization).lower()
        if initialization == "sobol":
            engine = torch.quasirandom.SobolEngine(self.dim, scramble=True, seed=int(self.cfg.seed))
            return engine.draw(int(self.cfg.num_particles)).to(device=self.device, dtype=torch.float32)
        if initialization == "random":
            return torch.rand(
                (self.cfg.num_particles, self.dim),
                generator=self.generator,
                device=self.device,
                dtype=torch.float32,
            )
        raise ValueError(f"Unknown PSO initialization: {self.cfg.initialization!r}")

    def observe(self, scores: torch.Tensor) -> bool:
        """Update personal/global bests from a maximization score vector."""

        scores = scores.to(device=self.device, dtype=torch.float32)
        if scores.shape != self.personal_best_scores.shape:
            raise ValueError(
                f"Expected score shape {tuple(self.personal_best_scores.shape)}, got {tuple(scores.shape)}"
            )

        finite_scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -torch.inf))
        improved = finite_scores > self.personal_best_scores
        self.personal_best_scores = torch.where(improved, finite_scores, self.personal_best_scores)
        self.personal_best_positions[improved] = self.positions[improved]

        best_score, best_index = torch.max(self.personal_best_scores, dim=0)
        global_improved = bool(best_score > self.global_best_score)
        if global_improved:
            self.global_best_score = best_score.detach().clone()
            self.global_best_position = self.personal_best_positions[int(best_index)].detach().clone()
            self.iterations_since_global_improvement = 0
        else:
            self.iterations_since_global_improvement += 1
        return global_improved

    def observe_particles(
        self,
        particle_ids: torch.Tensor,
        positions: torch.Tensor,
        scores: torch.Tensor,
    ) -> bool:
        """Update personal/global bests from completed asynchronous particle rollouts."""

        particle_ids = particle_ids.to(device=self.device, dtype=torch.long)
        positions = positions.to(device=self.device, dtype=torch.float32)
        scores = scores.to(device=self.device, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] != self.dim:
            raise ValueError(f"Expected positions shape [N, {self.dim}], got {tuple(positions.shape)}")
        if particle_ids.shape != scores.shape or particle_ids.shape[0] != positions.shape[0]:
            raise ValueError("particle_ids, positions, and scores must have the same leading dimension.")
        if torch.any((particle_ids < 0) | (particle_ids >= self.positions.shape[0])):
            raise ValueError("particle_ids contains values outside the swarm range.")

        global_improved = False
        finite_scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -torch.inf))
        for local_index in range(int(particle_ids.numel())):
            particle_id = int(particle_ids[local_index].detach().cpu())
            score = finite_scores[local_index]
            if score > self.personal_best_scores[particle_id]:
                self.personal_best_scores[particle_id] = score
                self.personal_best_positions[particle_id] = positions[local_index]
                if score > self.global_best_score:
                    self.global_best_score = score.detach().clone()
                    self.global_best_position = positions[local_index].detach().clone()
                    global_improved = True

        if global_improved:
            self.iterations_since_global_improvement = 0
        return global_improved

    def inertia_for_iteration(self, total_iterations: int | None = None) -> float:
        """Return the configured inertia for the current iteration."""

        if self.cfg.inertia_start is None or self.cfg.inertia_end is None:
            return float(self.cfg.inertia)
        total = max(1, int(total_iterations or self.cfg.iterations) - 1)
        alpha = min(1.0, max(0.0, float(self.iteration) / float(total)))
        return float(self.cfg.inertia_start) + alpha * (float(self.cfg.inertia_end) - float(self.cfg.inertia_start))

    def _social_best_positions(self) -> torch.Tensor:
        topology = str(self.cfg.topology).lower()
        if topology == "global":
            return self.global_best_position.unsqueeze(0).expand_as(self.positions)
        if topology == "ring":
            return ring_best_positions(
                self.personal_best_positions,
                self.personal_best_scores,
                max(1, int(self.cfg.neighborhood_size)),
            )
        raise ValueError(f"Unknown PSO topology: {self.cfg.topology!r}")

    def step(self, *, total_iterations: int | None = None) -> None:
        """Advance particle positions by one PSO update."""

        r1 = torch.rand(self.positions.shape, generator=self.generator, device=self.device)
        r2 = torch.rand(self.positions.shape, generator=self.generator, device=self.device)
        self.positions, self.velocities = pso_integrate_step(
            self.positions,
            self.velocities,
            self.personal_best_positions,
            self._social_best_positions(),
            r1,
            r2,
            self.inertia_for_iteration(total_iterations),
            float(self.cfg.cognitive),
            float(self.cfg.social),
            float(self.cfg.velocity_clamp),
        )
        self.iteration += 1

    def step_particles(self, particle_ids: torch.Tensor, *, total_iterations: int | None = None) -> None:
        """Advance selected particles by one bounded PSO update."""

        if particle_ids.numel() == 0:
            return
        particle_ids = torch.unique(particle_ids.to(device=self.device, dtype=torch.long))
        social_best_positions = self._social_best_positions()[particle_ids]
        r1 = torch.rand((particle_ids.numel(), self.dim), generator=self.generator, device=self.device)
        r2 = torch.rand((particle_ids.numel(), self.dim), generator=self.generator, device=self.device)
        new_positions, new_velocities = pso_integrate_step(
            self.positions[particle_ids],
            self.velocities[particle_ids],
            self.personal_best_positions[particle_ids],
            social_best_positions,
            r1,
            r2,
            self.inertia_for_iteration(total_iterations),
            float(self.cfg.cognitive),
            float(self.cfg.social),
            float(self.cfg.velocity_clamp),
        )
        self.positions[particle_ids] = new_positions
        self.velocities[particle_ids] = new_velocities
        self.async_updates += int(particle_ids.numel())

    def maybe_restart_stagnant_particles(self) -> int:
        """Restart a fraction of low-performing particles after global-best stagnation."""

        restart_after = int(self.cfg.restart_after_iterations)
        restart_fraction = float(self.cfg.restart_fraction)
        if restart_after <= 0 or restart_fraction <= 0.0:
            return 0
        if self.iterations_since_global_improvement < restart_after:
            return 0

        num_particles = int(self.positions.shape[0])
        if num_particles <= 1:
            return 0
        restart_count = max(1, min(num_particles - 1, int(round(num_particles * restart_fraction))))
        if restart_count <= 0:
            return 0

        _scores, restart_ids = torch.topk(self.personal_best_scores, k=restart_count, largest=False)
        self.positions[restart_ids] = torch.rand(
            (restart_count, self.dim),
            generator=self.generator,
            device=self.device,
            dtype=torch.float32,
        )
        vel_scale = float(self.cfg.velocity_clamp)
        self.velocities[restart_ids] = (
            2.0 * torch.rand((restart_count, self.dim), generator=self.generator, device=self.device) - 1.0
        ) * vel_scale
        if bool(self.cfg.reset_personal_best_on_restart):
            self.personal_best_positions[restart_ids] = self.positions[restart_ids]
            self.personal_best_scores[restart_ids] = -torch.inf
        self.iterations_since_global_improvement = 0
        return restart_count

    def blend_global_best_score(self, reevaluated_score: torch.Tensor) -> None:
        """Blend a noisy re-evaluation into the stored global-best score."""

        if not torch.isfinite(reevaluated_score):
            return
        blend = min(1.0, max(0.0, float(self.cfg.best_reevaluate_blend)))
        score = reevaluated_score.to(device=self.device, dtype=torch.float32)
        if torch.isfinite(self.global_best_score):
            self.global_best_score = (1.0 - blend) * self.global_best_score + blend * score
        else:
            self.global_best_score = score.detach().clone()

    def state_dict(self) -> dict:
        return {
            "cfg": {
                "num_particles": int(self.cfg.num_particles),
                "topology": str(self.cfg.topology),
                "neighborhood_size": int(self.cfg.neighborhood_size),
            },
            "dim": self.dim,
            "positions": self.positions,
            "velocities": self.velocities,
            "personal_best_positions": self.personal_best_positions,
            "personal_best_scores": self.personal_best_scores,
            "global_best_position": self.global_best_position,
            "global_best_score": self.global_best_score,
            "iteration": self.iteration,
            "async_updates": self.async_updates,
            "iterations_since_global_improvement": self.iterations_since_global_improvement,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["dim"]) != self.dim:
            raise ValueError(f"Checkpoint dim {state['dim']} does not match current dim {self.dim}")
        if tuple(state["positions"].shape) != tuple(self.positions.shape):
            raise ValueError(
                f"Checkpoint position shape {tuple(state['positions'].shape)} does not match current shape "
                f"{tuple(self.positions.shape)}"
            )
        for name in (
            "positions",
            "velocities",
            "personal_best_positions",
            "personal_best_scores",
            "global_best_position",
            "global_best_score",
        ):
            setattr(self, name, state[name].to(self.device))
        self.iteration = int(state["iteration"])
        self.async_updates = int(state.get("async_updates", 0))
        self.iterations_since_global_improvement = int(state.get("iterations_since_global_improvement", 0))
        generator_state = state.get("generator_state")
        if generator_state is not None:
            if isinstance(generator_state, torch.Tensor):
                generator_state = generator_state.to(device="cpu", dtype=torch.uint8)
            else:
                generator_state = torch.as_tensor(generator_state, device="cpu", dtype=torch.uint8)
            self.generator.set_state(generator_state)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        # Older local checkpoints stored the SwarmConfig dataclass. PyTorch 2.6
        # defaults torch.load to weights_only=True, which rejects such objects.
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.load_state_dict(state)
