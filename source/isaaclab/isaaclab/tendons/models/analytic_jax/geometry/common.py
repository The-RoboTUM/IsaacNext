# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class TendonLengthOutput:
    """Result of one tendon length calculation."""

    delta_l: jnp.ndarray
    length: jnp.ndarray | None = None
    state: dict[str, jnp.ndarray] | None = None
    debug: dict[str, Any] | None = None


@dataclass
class TendonDeltaLengths:
    """All five spring-length deltas used by the analytic tendon model."""

    gst: jnp.ndarray
    dft: jnp.ndarray
    kft: jnp.ndarray
    edt1: jnp.ndarray
    edt2: jnp.ndarray
    debug: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, jnp.ndarray]:
        return {
            "gst": self.gst,
            "dft": self.dft,
            "kft": self.kft,
            "edt1": self.edt1,
            "edt2": self.edt2,
        }

    def as_tuple(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.gst, self.dft, self.kft, self.edt1, self.edt2


def angle_from_sws(a: jnp.ndarray, b: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Angle helper used throughout the tendon geometry equations.

    This is deliberately TorchScript-compatible so the debug and JIT paths use
    the exact same helper implementation.
    """
    x = a - b * jnp.cos(theta)
    y = b * jnp.sin(theta)
    return jnp.arctan2(y, x)
