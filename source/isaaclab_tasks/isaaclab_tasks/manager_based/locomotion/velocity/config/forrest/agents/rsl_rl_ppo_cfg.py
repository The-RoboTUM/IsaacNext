# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.tendons.parameter_loader import load_forrest_parameter_config
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

FORREST_PARAMS = load_forrest_parameter_config()
AGENT_PARAMS = FORREST_PARAMS.agent


@configclass
class ForrestRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = AGENT_PARAMS.runner.seed
    device = AGENT_PARAMS.runner.device
    num_steps_per_env = AGENT_PARAMS.runner.num_steps_per_env
    max_iterations = AGENT_PARAMS.runner.max_iterations
    save_interval = AGENT_PARAMS.runner.save_interval
    experiment_name = AGENT_PARAMS.runner.experiment_name
    run_name = AGENT_PARAMS.runner.run_name
    logger = AGENT_PARAMS.runner.logger
    neptune_project = AGENT_PARAMS.runner.neptune_project
    wandb_project = AGENT_PARAMS.runner.wandb_project
    empirical_normalization = AGENT_PARAMS.runner.empirical_normalization
    clip_actions = AGENT_PARAMS.runner.clip_actions
    check_for_nan = AGENT_PARAMS.runner.check_for_nan
    resume = AGENT_PARAMS.runner.resume
    load_run = AGENT_PARAMS.runner.load_run
    load_checkpoint = AGENT_PARAMS.runner.load_checkpoint
    if AGENT_PARAMS.runner.obs_groups is not None:
        obs_groups = AGENT_PARAMS.runner.obs_groups
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=AGENT_PARAMS.policy.init_noise_std,
        noise_std_type=AGENT_PARAMS.policy.noise_std_type,
        state_dependent_std=AGENT_PARAMS.policy.state_dependent_std,
        actor_obs_normalization=AGENT_PARAMS.policy.actor_obs_normalization,
        critic_obs_normalization=AGENT_PARAMS.policy.critic_obs_normalization,
        actor_hidden_dims=AGENT_PARAMS.policy.actor_hidden_dims,
        critic_hidden_dims=AGENT_PARAMS.policy.critic_hidden_dims,
        activation=AGENT_PARAMS.policy.activation,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=AGENT_PARAMS.algorithm.value_loss_coef,
        use_clipped_value_loss=AGENT_PARAMS.algorithm.use_clipped_value_loss,
        clip_param=AGENT_PARAMS.algorithm.clip_param,
        entropy_coef=AGENT_PARAMS.algorithm.entropy_coef,
        num_learning_epochs=AGENT_PARAMS.algorithm.num_learning_epochs,
        num_mini_batches=AGENT_PARAMS.algorithm.num_mini_batches,
        learning_rate=AGENT_PARAMS.algorithm.learning_rate,
        schedule=AGENT_PARAMS.algorithm.schedule,
        gamma=AGENT_PARAMS.algorithm.gamma,
        lam=AGENT_PARAMS.algorithm.lam,
        desired_kl=AGENT_PARAMS.algorithm.desired_kl,
        max_grad_norm=AGENT_PARAMS.algorithm.max_grad_norm,
        optimizer=AGENT_PARAMS.algorithm.optimizer,
        normalize_advantage_per_mini_batch=AGENT_PARAMS.algorithm.normalize_advantage_per_mini_batch,
    )


@configclass
class ForrestFlatPPORunnerCfg(ForrestRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.max_iterations = 1500
        self.experiment_name = AGENT_PARAMS.runner.flat_experiment_name
        # self.policy.actor_hidden_dims = [256, 128, 128]
        # self.policy.critic_hidden_dims = [256, 128, 128]
