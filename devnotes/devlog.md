# devlog

## 02.07

- Added a live Forrest tendon calibration mode to `scripts/tendons/run.py` with `--calibration`, including docked
  Kit UI controls, controller/tendon/baseline parameter tabs, live range editing, reset/pause/stop controls, and
  live plots for controller commands and tendon torques.
- Added a viewport tendon overlay for calibration runs. The overlay reuses the analytic tendon visualization path
  construction, draws each tendon in a distinct color, and highlights active tendons with brighter/thicker lines.
- Kept calibration physics aligned with normal replay: the calibration controller bridge now uses the same open-loop
  controller classes as the non-calibration runner, respects `include_knee`, and avoids extra active-loop Kit ticks.
- Promoted the latest PSO-selected CPG and tendon-length parameters into the default Forrest profile while keeping
  previous values commented in the YAML for reference.
- Moved ground and foot material settings into the Forrest parameter profile and wired them into standalone replay,
  PSO, and RL training so rubber-foot friction is configured consistently across workflows.
- Extended Forrest parameter loading/export to cover training material settings and tendon stiffness parameters.
- Updated PSO to support the simpler CPG controller search, replayable `best.yaml` exports, raw/terminated velocity
  diagnostics, more tolerant unphysical terminations, and safer contact-sensor body handling.
- Added or extended focused tests around Forrest parameter loading, curriculum behavior, and CPG/replay consistency.

## 01.07

- *NOTE TO READER:* Roadmap of concrete todos to implement IsaacNext -> Identix data collection can be found in
  `todos_for_data_recording.md`
- Review implementation of points 1 to 4 (especially unchecked TODOs, since some of them are small fixes to current
  bugs).

## 30.06

- Started the Identix data-recording implementation for the Forrest tendon-chain dataset.
- Added the first 5-DOF state contract for left-leg `sim_data`: `l3f_femorotibial_front`,
  `l4f_intertarsal_front`, `l5_metatarsophalangeal`, `l6_interphalangeal`, and `l8_knee_flexor`.
- Added `isaaclab.tendons.data_recording.DataRecording` and `DataRecordingConfig`.
- The recorder writes Identix-compatible `sim_data` columns in positional order:
  `q0..q4`, `dq0..dq4`, `ddq0..ddq4`, `tau0..tau4`.
- Added separate `sample_context` rows for time/env/side and optional `spatial_data` rows for world-frame root/body
  diagnostics, keeping projection out of the recorder.
- Added metadata output with resolved joint/body mappings, omitted first-pass joints, sim settings, tau source, and row
  count.
- Verified the new recorder with `py_compile` and a fake-robot SQLite smoke test in the `next` conda environment.
- Wired the recorder into `scripts/tendons/run.py` behind opt-in `--record_identix` flags. Recording remains disabled by
  default, writes post-step state, and prints the SQLite/metadata paths at shutdown.
- Added `scripts/tendons/validate_identix_recording.py` to check the tiny `sim_data` contract: exact column order,
  nonempty rows, finite values, metadata consistency, units, finite-difference residuals, column statistics, spatial
  diagnostics, and optional Identix `SystemDataset` loading.
- Validated the validator with a deterministic fake recording under `/tmp/identix_recorder_validation`.
- Next real IsaacNext smoke test:
  `./isaaclab.sh -p scripts/tendons/run.py --headless --constraint_mode static --controller sin --duration 1.0 --record_identix --record_output_dir outputs/identix_recording_tiny_static --record_overwrite --record_side left --record_joint_set tendon_chain_5 --record_spatial_state`.
- Validate the generated recording with:
  `conda run --no-capture-output -n next python scripts/tendons/validate_identix_recording.py outputs/identix_recording_tiny_static`.
- Optional Identix loader check, using an environment with Identix dependencies:
  `conda run --no-capture-output -n test python scripts/tendons/validate_identix_recording.py outputs/identix_recording_tiny_static --check_identix`.
- Ran a very short real IsaacSim smoke recording into `/tmp/identix_recording_real_smoke` with static base, left leg,
  non-JIT debug mode, spatial diagnostics enabled, and `applied_torque` as the `sim_data.tau` source.
- The smoke recording validated successfully: 20 `sim_data` rows, 20 `sample_context` rows, 120 `spatial_data` rows,
  finite values, expected column order/units, and successful `SystemDataset(num_dofs=5)` loading from the `test` env.
- The escalated IsaacSim process did not return cleanly through the tool session after the 0.05 s smoke run, so it was
  terminated by matching the unique `/tmp/identix_recording_real_smoke` argument. The generated recording was complete
  and validated.

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
