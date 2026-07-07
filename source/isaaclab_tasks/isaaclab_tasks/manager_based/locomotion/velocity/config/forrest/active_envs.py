# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .base import ForrestBaseEnvCfg
from .boom import BoomConstraintCfg, create_planar_boom_constraint
from .rl_env_cfg import FORREST_PARAMS, TRAINING_PARAMS


class BoomMixin:
    """Adds a sagittal-plane physical boom constraint to a Forrest env."""

    def __post_init__(self):
        super().__post_init__()

        # The boom authors per-env USD joints before startup; replicated physics would share those properties.
        self.scene.replicate_physics = False

        pose_range = dict(self.events.reset_base.params["pose_range"])
        pose_range["y"] = (0.0, 0.0)
        pose_range["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = pose_range
        if self.events.startup_reset_base is not None:
            self.events.startup_reset_base.params["pose_range"] = pose_range

        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.events.create_planar_boom = EventTerm(
            func=create_planar_boom_constraint,
            mode="prestartup",
            params={"cfg": _boom_constraint_cfg()},
        )


def _boom_constraint_cfg() -> BoomConstraintCfg:
    boom = FORREST_PARAMS.boom
    return BoomConstraintCfg(
        body_path_template=boom.body_path_template,
        joint_path_template=boom.joint_path_template,
        locked_axes=tuple(boom.locked_axes),
        lock_x_angle=boom.lock_x_angle,
        body_anchor_pos=tuple(boom.body_anchor_pos),
        body_anchor_rot_wxyz=tuple(boom.body_anchor_rot_wxyz),
        debug=boom.debug,
    )


@configclass
class ForrestRoughEnvCfg(ForrestBaseEnvCfg):
    """Forrest locomotion on generated rough terrain."""

    pass


@configclass
class ForrestFlatEnvCfg(ForrestBaseEnvCfg):
    """Forrest locomotion on a flat plane."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class ForrestRoughBoomEnvCfg(BoomMixin, ForrestRoughEnvCfg):
    """Forrest rough-terrain env constrained to the sagittal plane by a boom."""

    pass


@configclass
class ForrestFlatBoomEnvCfg(BoomMixin, ForrestFlatEnvCfg):
    """Forrest flat-terrain env constrained to the sagittal plane by a boom."""

    pass


@configclass
class ForrestRoughEnvCfg_PLAY(ForrestRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = TRAINING_PARAMS.play_episode_length_s
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class ForrestFlatEnvCfg_PLAY(ForrestFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class ForrestRoughBoomEnvCfg_PLAY(ForrestRoughBoomEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = TRAINING_PARAMS.play_episode_length_s
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class ForrestFlatBoomEnvCfg_PLAY(ForrestFlatBoomEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
