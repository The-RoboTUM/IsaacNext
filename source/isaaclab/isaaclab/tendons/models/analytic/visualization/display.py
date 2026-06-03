"""Small formatting helpers for readable side panels."""

from __future__ import annotations

import numpy as np


def active_color(delta_l: float) -> str:
    """Color convention used throughout the visualizer."""
    return "grey" if delta_l > 0 else "green"


def mm_text(value_m: float, width: int = 7, precision: int = 3) -> str:
    """Format meters as millimeters for compact debug panels."""
    return f"{value_m * 1000:{width}.{precision}f} mm"


def deg_text(value_rad: float, width: int = 6, precision: int = 1) -> str:
    """Format radians as degrees for compact debug panels."""
    return f"{np.rad2deg(value_rad):{width}.{precision}f}°"


def bool_text(value: bool) -> str:
    """Compact boolean formatting for debug panels."""
    return "yes" if bool(value) else "no"


def table_lines(rows: list[tuple[str, object]], *, key_width: int = 12) -> str:
    """Return a monospace-friendly key/value table."""
    return "\n".join(f"{key:<{key_width}} {value}" for key, value in rows)
