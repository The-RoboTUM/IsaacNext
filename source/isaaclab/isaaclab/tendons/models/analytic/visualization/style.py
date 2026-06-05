# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matplotlib styling helpers for tendon visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
from datetime import datetime

try:
    import seaborn as sns
except ImportError:  # keep the script usable if seaborn is not installed
    sns = None

ACTIVE_TENDON_STYLE = {"color": "green", "linestyle": "-", "linewidth": 3.0}
INACTIVE_TENDON_STYLE = {"color": "red", "linestyle": "--", "linewidth": 1.5}
SINGLE_PLOT_TENDON_STYLES = {
    "gst_upper": {"inactive": "tab:red", "active": "tab:green"},
    "gst_lower": {"inactive": "tab:pink", "active": "limegreen"},
    "kft": {"inactive": "tab:purple", "active": "darkviolet"},
    "dft": {"inactive": "tab:orange", "active": "darkorange"},
    "edt1": {"inactive": "tab:blue", "active": "navy"},
    "edt2": {"inactive": "tab:brown", "active": "saddlebrown"},
}


def log(message: str):
    """Print a timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def configure_plot_style():
    """Configure plotting defaults without changing the geometry code."""
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "mathtext.fontset": "dejavusans",
    })


def rest_length_label(name: str, length_m: float) -> str:
    """Legend label with static tendon rest length."""
    return rf"{name} ($L_0={length_m * 1000:.1f}\,\mathrm{{mm}}$)"


def set_tendon_active_style(line, active: bool, tendon_name: str = "default", single_plot: bool = False):
    """Show active tendons as solid and thicker; use paired colors in single-plot mode."""
    if single_plot and tendon_name in SINGLE_PLOT_TENDON_STYLES:
        colors = SINGLE_PLOT_TENDON_STYLES[tendon_name]
        line.set_color(colors["active"] if active else colors["inactive"])
        line.set_linestyle("-" if active else "--")
        line.set_linewidth(3.2 if active else 1.4)
        return

    style = ACTIVE_TENDON_STYLE if active else INACTIVE_TENDON_STYLE
    line.set_color(style["color"])
    line.set_linestyle(style["linestyle"])
    line.set_linewidth(style["linewidth"])
