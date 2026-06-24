# RL Experiment Tracking

RSL-RL training has a small experiment-tracking layer that stays disabled by default. It writes local metadata for every run and can attach to the existing RSL-RL Weights & Biases logger when `--logger=wandb` is selected.

## Local Files

Each training run stores:

```text
logs/rsl_rl/<experiment>/<run>/
├── model_<iteration>.pt
├── latest.pt
├── final.pt
├── params/
│   ├── agent.yaml
│   └── env.yaml
└── tracking/
    ├── run_metadata.json
    └── git_diff.patch
```

`run_metadata.json` includes the resolved environment and agent config, seed, git commit and dirty status, hostname, GPU name, Python/PyTorch/CUDA/Isaac Lab/RSL-RL versions, Forrest config path/hash, and run start time.

## W&B

Install dependencies through the repository setup or update an existing environment:

```bash
conda activate sim
pip install "wandb>=0.17"
wandb login
```

Start training with W&B:

```bash
./isaaclab.sh \
 -p scripts/reinforcement_learning/rsl_rl/train.py \
 --task=Isaac-Velocity-Rough-Forrest-v0 \
 --headless \
 --logger=wandb \
 --log_project_name=isaacnext-forrest
```

If `wandb` is unavailable, training falls back to local files/TensorBoard instead of failing.

## Frequency Controls

Extra tracking work is frequency-gated so normal training stays close to the native RSL-RL path. The most useful knobs are:

```bash
--tracking_extra_metrics_interval=1
--tracking_raw_reward_interval=10
--tracking_tendon_metrics_interval=10
--tracking_checkpoint_alias_interval=1
--tracking_checkpoint_artifact_interval=0
```

Set an interval to `0` to disable that feature. W&B checkpoint artifacts are disabled by default because uploading large checkpoint files can block training; local `latest.pt` and `final.pt` aliases remain enabled by default.

## Metrics

The tracking layer logs stable namespaces where the data is available:

```text
train/return_mean
train/episode_length_mean
train/env_steps
train/fps

ppo/policy_loss
ppo/value_loss
ppo/entropy
ppo/learning_rate

reward/raw/<reward_term>
reward/weighted/<reward_term>
termination/<termination_reason>

tendon/tension_mean
tendon/tension_p95
tendon/tension_max
tendon/slack_fraction
```

RSL-RL 5.0.1 does not expose approximate KL or clip fraction in its public loss dictionary, so those metrics are not fabricated. Tendon saturation is also omitted unless a real saturation limit is added to the tendon model.

## Resume

Use `latest.pt` for the usual resume path:

```bash
./isaaclab.sh \
 -p scripts/reinforcement_learning/rsl_rl/train.py \
 --task=Isaac-Velocity-Rough-Forrest-v0 \
 --headless \
 --resume \
 --load_run=<run-folder-name> \
 --checkpoint=latest.pt
```

Checkpoints keep the native RSL-RL model, optimizer, normalizer, and iteration state. The tracking hook also stores available RNG state and environment-step count in checkpoint `infos`.

## Evaluation

Run deterministic evaluation independently of training:

```bash
./isaaclab.sh \
 -p scripts/reinforcement_learning/rsl_rl/evaluate.py \
 --task=Isaac-Velocity-Flat-Forrest-Play-v0 \
 --num_envs=64 \
 --episodes=128 \
 --load_run=<run-folder-name> \
 --checkpoint=latest.pt \
 --lin_vel_x=0.5 \
 --lin_vel_y=0.0 \
 --ang_vel_z=0.0
```

Results are written as JSON under `logs/rsl_rl/<experiment>/<run>/eval/` unless `--output` is set. The summary includes return, episode length, termination counts, command tracking error when available, and tendon safety metrics when available.

## Profiling

Profiling is opt-in and disabled during normal training. Enable the lightweight wall-clock profiler with:

```bash
./isaaclab.sh \
 -p scripts/reinforcement_learning/rsl_rl/train.py \
 --task=Isaac-Velocity-Rough-Forrest-v0 \
 --headless \
 --num_envs=4096 \
 --profile
```

The profiler writes:

```text
logs/rsl_rl/<experiment>/<run>/profile/profile_summary.json
```

The summary contains call counts, total time, mean time, and max time for the main training phases: policy action, environment step, RSL-RL step processing, PPO update, logging, and checkpoint saves.

For manager-based environments, the same JSON also breaks down `env.step`:

```text
env/action_process
env/physics_total
env/physics/action_apply
env/physics/write_data_to_sim
env/physics/sim_step
env/physics/scene_update
env/termination
env/reward
env/reset
env/command_event
env/observation
env/tendon/apply_jit
tendon/joint_angles
tendon/delta_lengths_jit
tendon/energy_jit
tendon/damping
tendon/autograd_torques
tendon/store_delta_lengths
tendon/compute_torques_jit
tendon/torque_mapping_jit
tendon/external_forces
tendon/set_wrenches
tendon/apply_jit_total
```

Use these fields to decide whether optimization should target PhysX stepping, tendon dynamics, reward/termination terms, observation construction, or reset logic. The `tendon/*` records split the passive tendon path into joint-state reads, JIT geometry/energy, damping, autograd torque computation, torque mapping, external force construction, and wrench submission.
