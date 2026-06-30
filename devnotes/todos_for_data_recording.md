# Identix Data Recording Todo

Goal: record IsaacNext Forrest simulation data that can be used to build Identix-compatible databases for a new,
higher-DOF Forrest leg system.

The old Identix tendon dataset is useful as a schema reference, but it is not the final model contract for this robot.
The first realistic Forrest dataset should use the existing IsaacNext tendon-chain joint set as a simpler bridge: five
angular coordinates per leg, recorded directly from IsaacSim. Data collection should be 3D: record world-frame spatial
diagnostics alongside the Identix-compatible joint-coordinate table. Any planar projection should happen only in
visualization.

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
- [x] Record the left leg first.
- [x] Keep the recorder design side-aware so both legs can be added later.
- [x] First-pass joint set: the five tendon-chain joints already used by the IsaacNext tendon debug path.
- [x] Treat the old Identix 3-DOF tendon model as a reference only. A new Identix system/config is needed.
- [x] Record values computed by IsaacSim/IsaacNext; do not generate first-pass databases from the old Identix synthetic
  system.
- [ ] Keep generated databases, videos, and plots out of git unless explicitly curated.

## Dataset Reference

Identix currently uses two database types:

- `sim_data`: main learning dataset. `SystemDataset` reads columns positionally as `q`, `dq`, `ddq`, and `tau`.
  For the first Forrest tendon-chain dataset, use `q0..q4`, `dq0..dq4`, `ddq0..ddq4`, `tau0..tau4`.
- `dynamics_data`: optional known-dynamics decomposition. It stores aligned rows with `sample_id`, `time`, `q`, `dq`,
  `ddq`, `tau`, and component terms such as inertia, coriolis, potential, friction, model, and residual torque.

For the first implementation pass, generate `sim_data` plus metadata and spatial diagnostics. Add `dynamics_data` only
after the `sim_data` recorder and visualization pass are working.

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

## 1. Define The 5-DOF State Contract

- [x] First-pass side policy: `left_only`.
- [x] Data-collection constraint mode: `static`.
- [x] Data-collection controller mode: `sin`.
- [x] Do not use `boom` or `static_boom` for the first production dataset.
- [x] First-pass left-leg 5-DOF chain order:
    - `l3f_femorotibial_front`
    - `l4f_intertarsal_front`
    - `l5_metatarsophalangeal`
    - `l6_interphalangeal`
    - `l8_knee_flexor`
- [x] First-pass right-leg 5-DOF chain order for later:
    - `r3f_femorotibial_front`
    - `r4f_intertarsal_front`
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
- [x] Do not include hip roll, hip lateral/yaw, pantograph, or fourbar helper joints in the first `sim_data` table.
- [x] Store omitted joint names in metadata so it is clear this is a reduced subsystem dataset.
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
    - joint set, initially `tendon_chain_5`
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
    - likely first choice: `robot.data.applied_torque[:, selected_joint_indices]`
    - also inspect: `robot.data.computed_torque[:, selected_joint_indices]`
    - do not mix torque channels silently
- [x] If `tau` semantics are still unresolved, write a clearly documented placeholder only for shape/load testing, and
  do not train force-based objectives from that placeholder.
- [x] Record 3D spatial diagnostics in a separate table/file:
    - root pose and velocity in world frame
    - selected body/link poses in world frame
    - selected body/link linear and angular velocities in world frame
    - selected body/link accelerations when available
    - incoming joint wrench is deferred to the force/dynamics recording pass
- [ ] Store non-selected torque channels in diagnostics, not in the Identix `sim_data` table.
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

- [ ] Add recording CLI flags to `scripts/tendons/run.py`, or create `scripts/tendons/record_identix.py` if the run
  script
  becomes cluttered.
- [ ] Keep recording disabled by default.
- [ ] Proposed CLI flags:
    - `--record_identix`
    - `--record_output_dir`
    - `--record_side left`
    - `--record_joint_set tendon_chain_5`
    - `--record_spatial_state`
    - `--record_body_set tendon_chain_links`
    - `--record_tau_source applied_torque`
    - `--record_stride`
    - `--record_start_time`
    - `--record_overwrite`
- [ ] First target command:
  `./isaaclab.sh -p scripts/tendons/run.py --constraint_mode static --duration 3.0 --record_identix --record_side left --record_joint_set tendon_chain_5 --record_spatial_state`.
- [ ] Later freefall comparison command:
  `./isaaclab.sh -p scripts/tendons/run.py --constraint_mode freefall --duration 3.0 --record_identix --record_side left --record_joint_set tendon_chain_5 --record_spatial_state`.
- [ ] Run without `--jit` first so the existing tendon debug path can provide diagnostics.
- [ ] Once `sim_data` is correct, allow the same recorder to run with `--jit` for faster collection when debug channels
  are
  not needed.
- [ ] Print database and metadata paths at the end of the run.

## 4. Generate And Validate Tiny `sim_data`

- [ ] Generate a tiny dataset first, for example 1-3 seconds.
- [ ] Write `sim_data` to SQLite.
- [ ] Write metadata to JSON or YAML next to the database.
- [ ] Store output under an ignored local path.
- [ ] Check the SQLite table exists and row count is nonzero.
- [ ] Check columns are exactly in Identix order.
- [ ] Check all recorded values are finite.
- [ ] Check units:
    - `q`: radians for joints.
    - `dq`: radians per second.
    - `ddq`: radians per second squared.
    - `tau`: documented generalized torque units.
- [ ] Check `dq ~= finite_difference(q)`.
- [ ] Check `ddq ~= finite_difference(dq)`.
- [ ] Print min, max, mean, and standard deviation for each column.
- [ ] Confirm all five chain joints have physically plausible ranges.
- [ ] Confirm 3D root/body trajectories are recorded in world-frame coordinates.
- [ ] Confirm no recorder step projects, zeros, or drops out-of-plane components.
- [ ] Load the dataset through `identix.data_manager.SystemDataset` with `num_dofs = 5`.

## 5. Build The Visualization Tool

- [ ] Use Identix `scripts/viz/viz_tendon.py`, `viz_hopper.py`, and `viz_jumper.py` as style/API references.
- [ ] Add a Forrest visualization script, likely `scripts/tendons/viz_recording.py` in IsaacNext or a matching
  `scripts/viz/viz_forrest_tendon_chain.py` in Identix.
- [ ] Read recorded SQLite `sim_data` and metadata.
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

- [ ] Decide the source of each force component:
    - `tau_inertia`
    - `tau_coriolis`
    - `tau_potential`
    - `tau_friction`
    - `tau_model`
    - `tau_residual`
- [ ] Inspect Isaac/PhysX APIs for each component:
    - generalized mass matrix for inertia
    - gravity compensation for gravitational potential contribution
    - available coriolis/centrifugal APIs, if any
    - actuator torque channels
    - tendon debug generalized torques
- [ ] Check whether tendon effects applied through `TendonManager` appear in any Isaac generalized torque tensor.
- [ ] If they do not, record tendon debug torques as a separate diagnostic/dynamics channel and document how they relate
  to `sim_data.tau`.
- [ ] Do not use the old Identix 3-DOF tendon system to fabricate the new 5-DOF dynamics database.
- [ ] If IsaacNext cannot expose a component cleanly, mark it unavailable in metadata rather than filling it with a
  misleading value.
- [ ] Ensure `dynamics_data` rows align one-to-one with `sim_data`.
- [ ] Include `sample_id` in the dynamics table.
- [ ] Add optional diagnostic columns only after the Identix-required columns are stable.
- [ ] Re-run tiny dataset validation with both `sim_data` and `dynamics_data`.

## 7. Train The LNN In Identix

- [ ] Add a new Identix parameter file for the 5-DOF Forrest tendon-chain dataset.
- [ ] Add or adapt a new system implementation in Identix/System Safari, for example `ForrestTendonChain5System`.
- [ ] Set `num_dofs = 5`.
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
- [ ] Add the full left-leg articulation state only after the 5-DOF chain dataset has passed Identix loading, training,
  and validation smoke tests.
- [ ] Reuse the recorder from PSO only after standalone recording is validated.
- [ ] Keep generated datasets out of git.

## Questions To Resolve

- [ ] Confirm which IsaacSim-computed torque channel should populate `sim_data.tau0..tau4`.
- [ ] Confirm whether tendon debug torques should be part of `tau`, `dynamics_data`, or diagnostics only.
- [ ] Confirm whether the first Identix training dataset should be 3D `freefall` or fixed-base 3D `static`.
- [ ] Confirm which 3D body/link set should be recorded for reconstruction and visualization.
- [ ] Confirm which Isaac/PhysX dynamics components are exposed reliably enough to populate `dynamics_data`.
- [ ] Confirm whether Identix should learn one side-independent leg model from left/right samples later, or a coupled
  two-leg model.
