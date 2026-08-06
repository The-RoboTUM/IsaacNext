# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config class for the tendon action term."""

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.tendons.models.analytic.constants import TendonConstantRandomizationRanges
from isaaclab.tendons.plugin.action_term import TendonActionTerm, TendonActionTermHybrid
from isaaclab.utils import configclass


@configclass
class TendonActionTermHybridCfg(ActionTermCfg):
    """Configuration for tendon-based action term."""

    class_type: type[ActionTerm] = TendonActionTermHybrid
    """The associated action term class.

    The class should inherit from :class:`isaaclab.managers.action_manager.ActionTerm`.
    """

    asset_name: str = "robot"
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """

    randomization_ranges: TendonConstantRandomizationRanges = TendonConstantRandomizationRanges()
    """Randomization ranges for tendon constants."""

    parameters_file: str | None = None
    """Optional centralized Forrest YAML path used to build tendon constants at runtime."""

    update_interval: int = 1
    """Passive tendon recompute interval in physics substeps. One preserves the current behavior."""

    model_type: str | None = None
    """Passive tendon model implementation. Supported values are ``analytic`` and ``identix_elastic``."""

    identix_bundle_dir: str | None = None
    """Path to a deployed Identix Forrest full-robot bundle when ``model_type`` is ``identix_elastic``."""

    identix_repo_path: str | None = None
    """Optional Identix checkout path used to import the deployment runtime lazily."""

    identix_compile: bool | None = None
    """Compile Identix/JAX deployment functions on first use."""

    identix_transfer: str | None = None
    """Torch/JAX tensor transfer mode for Identix inference: ``auto``, ``dlpack``, or ``numpy``."""

    identix_force_scale: float | None = None
    """Multiplier applied to Identix elastic force output before Isaac wrench mapping."""

    identix_force_sign: float | None = None
    """Sign applied to Identix elastic force output before Isaac wrench mapping."""

    identix_apply_mode: str | None = None
    """How to apply Identix generalized forces: ``joint_effort`` or ``link_wrench``."""


@configclass
class TendonActionTermCfg(ActionTermCfg):
    """Configuration for tendon-based action term."""

    class_type: type[ActionTerm] = TendonActionTerm
    """The associated action term class.

    The class should inherit from :class:`isaaclab.managers.action_manager.ActionTerm`.
    """

    asset_name: str = "robot"
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """

    randomization_ranges: TendonConstantRandomizationRanges = TendonConstantRandomizationRanges()
    """Randomization ranges for tendon constants."""

    parameters_file: str | None = None
    """Optional centralized Forrest YAML path used to build tendon constants at runtime."""
