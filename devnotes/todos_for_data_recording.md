# Identix Data Recording Todo

Goal: record IsaacNext Forrest simulation data that can be used to build Identix-compatible databases for a new,
higher-DOF Forrest leg system.

The old Identix tendon dataset is useful as a schema reference, but it is not the final model contract for this robot.
The first realistic Forrest dataset should record all real one-leg articulation joints, actuated and passive, directly
from IsaacSim while keeping the Identix positional column format. Data collection should be 3D: record world-frame
spatial diagnostics alongside the Identix-compatible joint-coordinate table. Any planar projection should happen only
in visualization.

The implementation order is:

1. Build the recorder and generate `sim_data`.
2. Build a visualization tool similar to the existing Identix `scripts/viz/*` tools.
3. Add force/dynamics recording and generate `dynamics_data`.
4. Train the LNN in Identix.
5. Validate learned results against held-out IsaacSim data from `sim_data`.

## Current Decisions

- [x] Work on branch `feature/identix-data-recording`.
- [x] Use the `next` conda environment for IsaacNext.
- [x] Use standalone `scripts/tendons/run.py` first, not `scripts/pso/run.py`.
- [x] Keep `static_boom` available for debugging, but do not use it as the default data-collection mode.
- [x] Use fixed-base 3D collection with `constraint_mode = static` for the first production dataset.
- [x] Keep `constraint_mode = freefall` as a later comparison once the fixed-base dataset works.
- [x] Avoid `boom` and `static_boom` for production datasets because they encode a planar constraint.
- [x] Remove the standalone virtual-ground spring from active tendon paths; use PhysX contact instead.
- [x] Record the left leg first.
- [x] Keep the recorder design side-aware so both legs can be added later.
- [x] First-pass joint set: all real joints for one leg, actuated and passive.
- [x] Treat the old Identix 3-DOF tendon model as a reference only. A new Identix system/config is needed.
- [x] Record values computed by IsaacSim/IsaacNext; do not generate first-pass databases from the old Identix synthetic
  system.
- [ ] Keep generated databases, videos, and plots out of git unless explicitly curated.
- [x] Write each run under `outputs/forrest_dbs_<datestamp>/`.
- [x] Write the Identix-style kinematics database inside that folder as `forrest_kinematics.db`.
- [x] Keep the kinematics SQLite database to one user table: `sim_data`.
- [x] Store mapping/settings metadata in sidecar JSON, not as extra SQLite tables.
- [x] Add `forrest_tendons.db` as a separate visualization/debug sidecar database.
- [x] Keep `forrest_tendons.db` to timed step data; store constants/static visualization variables in `viz_vars.json`.

## Dataset Reference

Identix currently uses two database types:

- `sim_data`: main learning dataset. `SystemDataset` reads columns positionally as `q`, `dq`, `ddq`, and `tau`.
  For the first Forrest all-real-joints dataset, use `q0..qN`, `dq0..dqN`, `ddq0..ddqN`, `tau0..tauN`, where `N`
  is determined by the recorded real joint list in metadata.
- `dynamics_data`: optional known-dynamics decomposition. It stores aligned rows with `sample_id`, `step_index`,
  `time`, `env_id`, `side`, and non-repeated component terms such as inertia, Coriolis/centrifugal, gravity, friction,
  model, and tendon-residual torque. The repeated `q`, `dq`, `ddq`, and motor-plus-ground `tau` values stay only in
  `sim_data`.

The current implementation generates `sim_data`, optional tendon visualization data, and `dynamics_data` sidecars
together under one timestamped recording directory.

## 0. Baseline Setup

- [x] Confirm this branch includes the latest `feature/tendons/pso` changes.
- [x] Confirm the setup environment is usable enough to run IsaacNext scripts.
- [x] Confirm the Forrest USD/symlink setup exists.
- [x] Run a short `scripts/tendons/run.py` smoke test before adding recording logic.
- [x] Add `static_boom` constraint mode for fixed base plus authored sagittal boom constraint.
- [ ] Re-run a short standalone fixed-base 3D tendon smoke test:
  `./isaaclab.sh -p scripts/tendons/run.py --constraint_mode static --controller sin --duration 1.0`.
- [ ] Keep a later freefall comparison smoke test:
  `./isaaclab.sh -p scripts/tendons/run.py --constraint_mode freefall --controller sin --duration 1.0`.

## 1. Define The Real One-Leg State Contract

- [x] First-pass side policy: `left_only`.
- [x] Data-collection constraint mode: `static`.
- [x] Data-collection controller mode: `sin`.
- [x] Do not use `boom` or `static_boom` for the first production dataset.
- [x] First-pass left-leg real joint order:
    - `lp1_pantograph`
    - `l0_acetabulofemoral_roll`
    - `l1_acetabulofemoral_lateral`
    - `l2_pseudo_acetabulofemoral_flexion`
    - `l3b_femorotibial_back`
    - `l3f_femorotibial_front`
    - `l4f_intertarsal_front`
    - `l4b_intertarsal_back`
    - `l4p_intertarsal_pulley`
    - `l5_metatarsophalangeal`
    - `l6_interphalangeal`
    - `l8_knee_flexor`
- [x] First-pass right-leg real joint order for later:
    - `rp1_pantograph`
    - `r0_acetabulofemoral_roll`
    - `r1_acetabulofemoral_lateral`
    - `r2_pseudo_acetabulofemoral_flexion`
    - `r3b_femorotibial_back`
    - `r3f_femorotibial_front`
    - `r4f_intertarsal_front`
    - `r4b_intertarsal_back`
    - `r4p_intertarsal_pulley`
    - `r5_metatarsophalangeal`
    - `r6_interphalangeal`
    - `r8_knee_flexor`
- [x] Validate selected joints against `robot.data.joint_names` at runtime.
- [x] Fail loudly if any selected joint is missing or duplicated.
- [x] Store the resolved mapping in metadata:
    - `q_index`
    - joint name
    - side
    - Isaac joint index
    - units
    - sign convention
    - offset convention
- [x] Include all real actuated and passive joints in the first `sim_data` table.
- [x] Exclude only non-real virtual/exporter artifacts from the first `sim_data` table.
- [x] Store omitted joint names in metadata so it is clear which virtual joints are outside the state.
- [x] Do not put root/base coordinates into the first `sim_data.q` table unless the Identix state contract is explicitly
  expanded to floating-base coordinates.
- [x] Record root/base and selected link 3D state as auxiliary spatial diagnostics.

## 2. Build The `sim_data` Recorder

- [x] Add `source/isaaclab/isaaclab/tendons/data_recording.py`.
- [x] Create a `DataRecording` class.
- [x] Keep recording logic independent from `scripts/tendons/run.py`.
- [x] Add a recorder config dataclass for:
    - output directory
    - SQLite filename
    - metadata filename
    - `sim_data` table name
    - joint set, initially `real_leg_joints`
    - side policy
    - selected joint names
    - selected 3D body/link names
    - whether to record 3D spatial state
    - sampling stride
    - startup skip duration
    - constraint mode
    - `sim_data` tau source
    - overwrite policy
- [x] Resolve selected joints once after `sim.reset()` and `robot.update(0.0)`.
- [x] Resolve selected 3D bodies/links once after `sim.reset()` and `robot.update(0.0)`.
- [x] Record `q` from `robot.data.joint_pos[:, selected_joint_indices]`.
- [x] Record `dq` from `robot.data.joint_vel[:, selected_joint_indices]`.
- [x] Record `ddq` from `robot.data.joint_acc[:, selected_joint_indices]`.
- [x] Document that IsaacLab computes `joint_acc` by finite differences of `joint_vel`.
- [x] For the first `sim_data`, choose one documented `tau` channel:
    - first training-label choice: controller actuator torque plus projected PhysX ground contact generalized torque
    - optional diagnostic source: `robot.data.applied_torque[:, selected_joint_indices]`
    - also inspect: `robot.data.computed_torque[:, selected_joint_indices]`
    - do not mix torque channels silently
- [x] Exclude tendon generalized forces from `sim_data.tau`; those are intended to be learned from the dynamics.
- [x] Document whether `tau` uses a placeholder or a physically meaningful label.
- [ ] Record 3D spatial diagnostics later in a separate diagnostics artifact, not in the Identix kinematics database:
    - root pose and velocity in world frame
    - selected body/link poses in world frame
    - selected body/link linear and angular velocities in world frame
    - selected body/link accelerations when available
    - incoming joint wrench is deferred to the force/dynamics recording pass
- [ ] Store non-selected torque channels in later diagnostics, not in the Identix `sim_data` table.
- [x] Write metadata with:
    - selected joint names and indices
    - spatial body/link set
    - spatial frame convention
    - side policy
    - joint set
    - constraint mode
    - controller
    - sim dt
    - startup hold settings
    - parameter file/profile
    - tau source used for `sim_data`
    - available tau sources
    - row count
- [x] Use buffered batch inserts rather than one SQLite transaction per physics step.
- [x] Keep projection settings out of the recorder.

## 3. Wire Recorder Into `run.py`

- [x] Add recording CLI flags to `scripts/tendons/run.py`, or create `scripts/tendons/record_identix.py` if the run
  script
  becomes cluttered.
- [x] Keep recording disabled by default.
- [x] Proposed CLI flags:
    - `--record_identix`
    - `--record_output_dir`
    - `--record_side left`
    - `--record_joint_set real_leg_joints`
    - `--record_spatial_state`
    - `--record_body_set tendon_chain_links`
    - `--record_tau_source controller_plus_ground`
    - `--record_stride`
    - `--record_start_time`
    - `--record_overwrite`
- [x] Move recording defaults into `configs/forrest/default/recording.yaml` while keeping CLI overrides.
- [x] Formatting of data output
    - [x] Use `.db` suffix for database files
    - [x] Use timestamped output directories named `outputs/forrest_dbs_<datestamp>/`
- [ ] Review sampling frequency of collected data
- [x] First target command is available:
  `./isaaclab.sh -p scripts/tendons/run.py --headless --constraint_mode static --controller sin --duration 1.0 --record_identix --record_overwrite --record_side left --record_joint_set real_leg_joints`.
- [x] Later freefall comparison command is available:
  `./isaaclab.sh -p scripts/tendons/run.py --headless --constraint_mode freefall --controller sin --duration 1.0 --record_identix --record_overwrite --record_side left --record_joint_set real_leg_joints`.
- [x] Run without `--jit` first so the existing tendon debug path can provide diagnostics.
- [x] Keep the same recorder path compatible with `--jit` for faster collection when debug channels
  are
  not needed.
- [ ] Validate the `--jit` recorder path after the non-JIT tiny dataset is correct.
- [x] Print database and metadata paths at the end of the run.

## 4. Generate And Validate Tiny `sim_data`

- [x] Add `scripts/tendons/validate_identix_recording.py` for post-run validation.
- [x] Validate the validator itself with a deterministic fake SQLite recording.
- [x] Generate a very short smoke dataset first.
- [ ] Generate a tiny 1-3 second dataset after the smoke dataset.
- [x] Write `sim_data` to SQLite.
- [x] Write metadata to JSON or YAML next to the database.
- [x] Store output under an ignored local path.
- [x] Check the SQLite table exists and row count is nonzero.
- [x] Check columns are exactly in Identix order.
- [x] Check all recorded values are finite.
- [x] Check units:
    - `q`: radians for joints.
    - `dq`: radians per second.
    - `ddq`: radians per second squared.
    - `tau`: documented generalized torque units.
- [x] Check `dq ~= finite_difference(q)`.
- [x] Check `ddq ~= finite_difference(dq)`.
- [x] Print min, max, mean, and standard deviation for each column.
- [x] Confirm all recorded real joints have physically plausible ranges.
- [x] Confirm 3D root/body trajectories are recorded in world-frame coordinates.
- [x] Confirm no recorder step projects, zeros, or drops out-of-plane components.
- [ ] Add optional loading through `identix.data_manager.SystemDataset` with `num_dofs` matching metadata.
- [ ] Load the real tiny IsaacSim dataset through `identix.data_manager.SystemDataset` with metadata-derived `num_dofs`.

## 5. Build The Visualization Tool

- [ ] Use Identix `scripts/viz/viz_tendon.py`, `viz_hopper.py`, and `viz_jumper.py` as style/API references.
- [x] Reuse and extend `scripts/tendons/draw_tendon_action.py` as the first Forrest visualization validator.
- [x] Preserve previous JSONL input and animation behavior.
- [x] Read `forrest_tendons.db` from a recording directory when present.
- [x] Fall back to trajectory-only playback from `forrest_kinematics.db` and metadata when no tendon DB is present.
- [x] Read recorded SQLite `sim_data` and metadata for trajectory-only playback.
- [ ] Read spatial diagnostics if available.
- [ ] Support projection options for display only:
    - `xy`
    - `xz`
    - `yz`
- [ ] Support a 3D view if practical.
- [ ] Plot/animate the selected tendon-chain links from recorded world-frame body/link positions.
- [ ] Overlay time-colored trajectories similar to the Identix visualizers.
- [ ] Add basic controls/options:
    - stride
    - max frames/path points
    - save path
    - video format/FPS
    - headless save mode
- [ ] Use the visualization to sanity-check tiny `sim_data` before collecting force data.

## 6. Add Force Recording And `dynamics_data`

- [x] Decide the source of each force component:
    - `tau_inertia`
    - `tau_coriolis`
    - `tau_gravity`
    - `tau_friction`
    - `tau_model`
    - `tau_tendon_residual`
- [x] Inspect Isaac/PhysX APIs for each component:
    - generalized mass matrix for inertia
    - gravity compensation for gravitational potential contribution
    - coriolis/centrifugal compensation forces
    - actuator torque channels
    - configured joint friction coefficients
- [x] Create `forrest_dynamics.db` together with `forrest_kinematics.db`.
- [x] Keep repeated `q`, `dq`, `ddq`, and `tau` out of `forrest_dynamics.db`.
- [x] Store one `dynamics_data` row per `sim_data` row, aligned by `sample_id`.
- [x] Store inverse-dynamics components as per-joint columns:
    - `tau_inertia*`
    - `tau_coriolis*`
    - `tau_gravity*`
    - `tau_friction*`
    - `tau_model*`
    - `tau_tendon_residual*`
- [ ] Validate the sign convention of `tau_tendon_residual = tau_model - sim_data.tau` against analytic tendon debug
  torques on a short run.
- [ ] Check whether tendon effects applied through `TendonManager` appear in any Isaac generalized torque tensor.
- [ ] If they do not, record tendon debug torques as a separate diagnostic/dynamics channel and document how they relate
  to `sim_data.tau`.
- [ ] Do not use the old Identix 3-DOF tendon system to fabricate the new all-real-joints dynamics database.
- [x] If IsaacNext cannot expose a component cleanly, mark it unavailable in metadata rather than filling it with a
  misleading value.
- [x] Ensure `dynamics_data` rows align one-to-one with `sim_data`.
- [x] Include `sample_id` in the dynamics table.
- [ ] Add optional diagnostic columns only after the Identix-required columns are stable.
- [ ] Re-run tiny dataset validation with both `sim_data` and `dynamics_data` on a machine with Isaac assets/GPU
  available.

## 7. Train The LNN In Identix

- [ ] Add a new Identix parameter file for the all-real-joints Forrest leg dataset.
- [ ] Add or adapt a new system implementation in Identix/System Safari, for example `ForrestRealLegJointsSystem`.
- [ ] Set `num_dofs` from the recorded joint metadata.
- [ ] Point Identix to the recorded IsaacNext `sim_data` database.
- [ ] Add `dynamics_data` only once force terms are trustworthy.
- [ ] Decide the first training objective:
    - kinematics/data-loader smoke test only
    - dynamics objective using `sim_data.tau`
    - potential/force objective using `dynamics_data`
- [ ] Run a data-loader smoke test before training.
- [ ] Run a tiny overfit/debug training job.
- [ ] Scale to a larger training split only after tiny overfit works.
- [ ] Save model outputs under an ignored or explicitly managed output path.

## 8. Validate Learned Results Against IsaacSim Data

- [ ] Reserve held-out IsaacSim trajectories from `sim_data` before training.
- [ ] Start validation from recorded initial states.
- [ ] Roll out the learned Identix/LNN model under the same recorded inputs/torques where possible.
- [ ] Compare learned rollout against IsaacSim records:
    - `q(t)`
    - `dq(t)`
    - optionally `ddq(t)`
    - optionally spatial reconstruction/projection when available
- [ ] Compare predicted force components against `dynamics_data` where available.
- [ ] Report quantitative errors:
    - per-DOF RMSE
    - rollout drift over time
    - energy/force residuals when available
- [ ] Produce validation plots similar to Identix validation outputs.
- [ ] Use the visualization tool to compare IsaacSim trajectories and learned rollouts.
- [ ] Only scale dataset duration/controller variety after validation passes on small held-out runs.

## 9. Scale Up And Reuse

- [ ] Increase dataset duration only after tiny-data validation passes.
- [ ] Add multiple controller profiles only after the schema is stable.
- [ ] Compare `freefall` and `static` 3D datasets before deciding what the first full training run should use.
- [ ] Add both-leg recording only after the left-leg dataset is validated.
- [x] Make the first dataset the full real left-leg articulation state instead of the reduced 5-DOF tendon chain.
- [x] Add opt-in Forrest database recording to `scripts/reinforcement_learning/rsl_rl/play.py`.
- [x] Keep RL training scripts free of recorder imports and per-step recorder checks.
- [x] Reuse the shared `DataRecording` writer for RL play output.
- [x] Support multi-env play by storing each selected env and side as separate samples.
- [x] Add `recording.env_ids` / `--record_env_ids` so large play batches can record only selected environments.
- [ ] Validate RL play recording on a GPU machine with Isaac assets available.
- [ ] Reuse the recorder from PSO only after standalone and RL play recording are validated.
- [ ] Keep generated datasets out of git.

## Questions To Resolve

- [x] Populate `sim_data.tau0..tauN` with controller actuator torques plus projected PhysX contact-sensor ground torques.
- [x] Do not use the standalone virtual-ground spring force as the ground-contact label.
- [x] Do not apply the standalone virtual-ground spring during Identix recording rollouts.
- [ ] Confirm whether tendon debug torques should be part of `tau`, `dynamics_data`, or diagnostics only.
- [ ] Confirm whether the first Identix training dataset should be 3D `freefall` or fixed-base 3D `static`.
- [ ] Confirm which 3D body/link set should be recorded for reconstruction and visualization.
- [ ] Confirm which Isaac/PhysX dynamics components are exposed reliably enough to populate `dynamics_data`.
- [ ] Confirm whether Identix should learn one side-independent leg model from left/right samples later, or a coupled
  two-leg model.
