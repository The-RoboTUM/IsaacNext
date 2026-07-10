# Forrest Dynamics Recording Handoff

Last updated: 2026-07-10

This note summarizes the current state of the Forrest dynamics database work so we can resume without re-deriving the same facts.

## Recording Command

The main recording path is the RL play script:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-Forrest-Play-v0 \
  --num_envs=100 \
  --record_forrest_dbs \
  --record_side left \
  --record_stride 1 \
  --record_max_steps 5000 \
  --record_overwrite \
  env.episode_length_s=4.0 \
  --headless
```

This goes through `ForrestRLRecorder` in:

```text
source/isaaclab/isaaclab/tendons/rl_recording.py
```

The shared SQLite writer and schema logic live in:

```text
source/isaaclab/isaaclab/tendons/data_recording.py
```

The default recording config is:

```text
configs/forrest/default/recording.yaml
```

Current relevant recording flags:

```yaml
record_dynamics: true
record_debug_dynamics: false
residual_filter_threshold: 100
dynamics_db_filename: forrest_dynamics.db
debug_dynamics_db_filename: debug.db
```

With `record_debug_dynamics: false`, only the compact production dynamics DB is exported. With `residual_filter_threshold: 100`, rows are filtered using the current diagnostic residual norm before export.

## Production Dynamics Convention

The production database is meant for Identix/sysid training with this convention:

```text
inertia + coriolis/centrifugal + gravity + tendon = external
external = actuation + contact + friction
```

The compact `forrest_dynamics.db` stores the 12 selected leg DOFs only:

```text
tau_inertia[0:12]
tau_coriolis[0:12]
tau_gravity[0:12]
tau_tendon[0:12]
tau_actuation[0:12]
tau_contact[0:12]
tau_friction[0:12]
tau_external[0:12]
```

The signs in the production DB are saved to match the equation above. Do not reinterpret these columns using the older debug residual equation without checking signs.

## Force Term Acquisition

### Inertia

`tau_inertia` is computed as selected leg rows of the full floating-base mass matrix times acceleration:

```text
M_leg,root * root_acc + M_leg,joint * joint_acc
```

For the RL recorder, the primary inertia currently uses raw IsaacLab/PhysX root and joint acceleration signals. A recording-interval finite-difference inertia variant was useful diagnostically but is not the production term.

Important lesson: considering the floating base matters. However, the measured `inertia_other_joints` contribution was effectively zero in the tested recordings, so other joint acceleration coupling did not explain the residual.

### Coriolis / Centrifugal

`tau_coriolis` is acquired from the PhysX generalized force APIs where available. The sign is converted into the actual generalized force convention used by the production equation.

Older compensation APIs printed deprecation warnings:

```text
getCoriolisAndCentrifugalCompensationForces
getGravityCompensationForces
```

The implementation should prefer the newer force APIs when available and fall back only when necessary.

### Gravity

`tau_gravity` is acquired from PhysX generalized gravity force APIs, again saved as an actual generalized force, not a compensation command.

### Tendon

`tau_tendon` is the generalized force induced by the tendon model. Two diagnostic variants were compared:

```text
tendon
tendon_model
```

They matched closely in the tested runs, so tendon projection itself was not the main residual source. The tendon term is on the left side of the production equation with gravity/coriolis because it behaves like a potential/bias term for our sysid objective.

### Actuation

`tau_actuation` in the compact production DB is intentionally conservative. PhysX implicit drives do not expose a clean explicit actuator torque in the same way an explicit torque actuator would.

Key diagnostic terms:

```text
actuation
actuation_command
actuation_estimated
actuation_estimated_hip
actuation_estimated_hip_lateral_flexion
drive_pd
drive_pd_clipped
```

Important lesson: the raw command/drive PD terms can be huge and are not automatically valid as physical generalized actuator forces. The hip-only estimated actuation explained part of the residual in short runs, but in larger/bad runs it was not stable enough to become a production force label.

If we need clean actuator labels, the best future direction is an explicit actuator variant where torques are applied directly and recorded exactly.

### Contact

`tau_contact` is projected from contact sensor forces and moments through the body Jacobians. The production term uses the validated contact projection:

```text
contact_validated = contact_force + contact_moment
```

but the moment contribution is discarded when the projected moment is clearly invalid:

```text
|contact_moment| / |contact_force| > 2
```

Debug-only contact breakdowns include:

```text
contact_force
contact_moment
contact_digit
contact_digit_force
contact_digit_moment
contact_connector
contact_connector_force
contact_connector_moment
contact_base
```

Important lesson: most contact was digit contact. Connector contact had rare spikes. Base contact was zero in the tested runs. Contact did not explain the worst residual bursts, which often happened with `contact = 0`.

### Friction, Damping, Armature

Joint friction is estimated from IsaacLab joint coefficients:

```text
friction_dynamic = -dynamic_friction * sign(dq)
friction_viscous = -viscous_friction * dq
tau_friction = friction_dynamic + friction_viscous
```

In the tested Forrest setup, these were effectively zero or too small to explain the residual.

Debug terms also include:

```text
drive_damping
armature_inertia
```

These did not explain the main residual either in the latest tests.

## Debug Database

When enabled:

```yaml
record_debug_dynamics: true
```

the recorder writes:

```text
debug.db
```

This contains the wide forensic schema with residual variants, solver diagnostics, contact groups, quality gates, drive terms, limit distances, and per-row issue flags.

The long terminal report is printed only in debug mode. Production runs should keep debug off unless actively investigating physics discrepancies.

## Residual Interpretation

The main debug residual used for filtering is the best current sysid diagnostic residual:

```text
hip23+internal
```

This is not the production force equation. It is a practical quality metric that includes useful diagnostic corrections for hip actuation and internal solver constraints.

Recent 500-row run summary:

```text
usable <= 100 N*m: 276 / 500 = 55.2%
usable <= 150 N*m: 408 / 500 = 81.6%
usable <= 200 N*m: 455 / 500 = 91.0%
```

Worst rows were dominated by huge inertia/command/solver spikes, especially in one environment around steps 50-67. These rows often had zero contact, so the issue was not missing ground-contact force.

Practical conclusion: the database can be useful for sysid if filtered. Treat unfiltered rows as mixed-quality data.

## What Lowers The Residual

Useful:

- Include floating-base inertia coupling.
- Use raw PhysX/IsaacLab root and joint acceleration for the primary inertia term.
- Validate contact moments and reject extreme projected moments.
- Record internal solver/limit diagnostics in debug mode.
- Filter by residual norm before exporting production training rows.
- Inspect per-environment quality; bad transient environments can dominate mean/max metrics.

Not very useful in the latest data:

- Other-joint inertia coupling. It was essentially zero.
- Contact group fitting as a primary explanation. Contact was not the main residual source.
- Joint friction, viscous damping, or armature. These were zero or too small.
- Full drive PD as a physical force label. It can become huge and is not clean with implicit PhysX drives.

## Known PhysX / IsaacLab Caveat

PhysX implicit drives, joint limits, contacts, and mimic/passive constraints are solved together. This means the true per-DOF forces are not always separable into clean independent physical labels after the fact.

This is why `debug.db` stores solver diagnostics separately instead of merging them into the production training columns. Solver forces are useful for quality assessment, but they should not be blindly treated as tendon, actuator, or contact labels.

## Recommended Next Steps

1. Run production recording with debug off and residual filtering on.

   Current config already does this:

   ```yaml
   record_debug_dynamics: false
   residual_filter_threshold: 100
   ```

2. After recording, inspect row counts in `forrest_kinematics.db` and `forrest_dynamics.db`.

   They should match after filtering.

3. For any suspicious production dataset, rerun a shorter debug recording:

   ```bash
   --record_debug_dynamics
   ```

   and optionally override filtering:

   ```bash
   --record_filter_residual_threshold 150
   ```

4. If residual quality is still too low, the highest-value future experiment is an explicit actuator recording mode, not more post-hoc fitting of implicit drive terms.

5. Keep the compact production DB clean. Put new forensic columns in `debug.db`, not `forrest_dynamics.db`.
