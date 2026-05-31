# isaaclab/tendons/__init__.py

__all__ = [
    "TendonManager",
    "AnalyticTendonEnergyModel",
]


def __getattr__(name):
    if name == "TendonManager":
        from isaaclab.tendons.manager import TendonManager

        return TendonManager

    if name == "AnalyticTendonEnergyModel":
        from isaaclab.tendons.models.analytic.analytic_energy_model import (
            AnalyticTendonEnergyModel,
        )

        return AnalyticTendonEnergyModel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
