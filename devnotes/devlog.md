# devlog

## 24.06

- Added a PSO workflow for tuning Forrest tendon and controller parameters, including `configs/pso.yaml`,
  `scripts/pso/run.py`, checkpoint/reload support, best-parameter export, and iteration metrics.
- Added PSO hyperparameter meta-tuning with Optuna through `configs/pso_meta.yaml` and `scripts/pso/meta_tune.py`.
- Implemented the PSO backend in `isaaclab.pso`: config validation, parameter-space mapping, Torch-based optimizer,
  batched Forrest evaluator, CPG kernel, logging helpers, and RL-style reward-weight scoring.
- Extended tendon tooling with reusable runner utilities, parameter loading/export support, tendon manager updates,
  and a CPG oscillator controller for open-loop gait exploration.
- Added command-bin curriculum support shared by RL and PSO, with binned velocity sampling, rolling attempt/success
  statistics, curriculum progress metrics, and RL command-generator integration.
- Updated Forrest RL training configuration to use the command curriculum, expose curriculum/bin metrics, and gate
  forward velocity tracking reward by actual forward progress.
- Added or updated Forrest locomotion rewards and self-collision configuration, including gait symmetry, foot
  contact/parallelism penalties, base-height tracking, and related training parameters.
- Added PSO diagnostics for rollout success/failure causes, including fall, unphysical termination, and backward
  progress termination.

## 05.06

- Merged newer Isaac Lab tooling direction and aligned the project around Ruff-based pre-commit.
- Added `scripts/setup_repo.sh` for post-clone setup: conda env, source install, hooks, and Forrest USD symlink.
- Cleaned root files by removing obsolete notes, old READMEs, debug output, and local test script.
- Removed tracked `saved_models/` artifacts from Git while keeping them ignored locally.
- Expanded `.gitignore` for local outputs, symlinks, caches, generated assets, and agent/IDE files.
- Fixed current Ruff/pre-commit issues in tendon scripts and constants.

## 04.06

- Added two new reward terms to the rough-terrain environment of forrest
- One function punishes the walking if the feet are crossed
- The second function punishes any contacts with the ground that do not happen parallel to the ground
