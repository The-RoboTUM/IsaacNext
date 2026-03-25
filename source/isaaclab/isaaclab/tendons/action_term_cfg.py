"""Config class for the tendon action term."""

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.tendons.action_term import TendonActionTerm, TendonActionTermHybrid
from isaaclab.tendons.constants import TendonConstantRandomizationRanges
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

    randomization_ranges: TendonConstantRandomizationRanges = (
        TendonConstantRandomizationRanges()
    )
    """Randomization ranges for tendon constants."""

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

    randomization_ranges: TendonConstantRandomizationRanges = (
        TendonConstantRandomizationRanges()
    )
    """Randomization ranges for tendon constants."""
