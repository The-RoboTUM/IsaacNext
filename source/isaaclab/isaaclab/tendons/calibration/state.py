# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared runtime state for live tendon calibration."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch

from isaaclab.tendons.controllers.base import DOF_ORDER


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata for one editable runtime parameter."""

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    group: str
    unit: str = ""


class CalibrationState:
    """Thread-tolerant state bridge between the sim loop and Kit UI callbacks."""

    def __init__(
        self,
        *,
        controller: str,
        controller_specs: dict[str, list[ParameterSpec]],
        tendon_specs: list[ParameterSpec],
        baseline_specs: list[ParameterSpec],
        max_plot_points: int = 600,
    ) -> None:
        self._lock = RLock()
        self.controller = controller
        self.paused = False
        self.reset_requested = False
        self.stop_requested = False
        self.controller_specs = controller_specs
        self.tendon_specs = tendon_specs
        self.baseline_specs = baseline_specs
        self.values: dict[str, float] = {}
        self.ranges: dict[str, tuple[float, float]] = {}
        self.telemetry: dict[str, Any] = {}
        self._tendon_rebuild_requested = False
        self.controller_history: deque[list[float]] = deque(maxlen=max_plot_points)
        self.tendon_history: deque[list[float]] = deque(maxlen=max_plot_points)

        for specs in controller_specs.values():
            for spec in specs:
                self.values[spec.name] = spec.default
                self.ranges[spec.name] = (spec.minimum, spec.maximum)
        for spec in tendon_specs:
            self.values[spec.name] = spec.default
            self.ranges[spec.name] = (spec.minimum, spec.maximum)
        for spec in baseline_specs:
            self.values[spec.name] = spec.default
            self.ranges[spec.name] = (spec.minimum, spec.maximum)

    def snapshot_values(self) -> dict[str, float]:
        with self._lock:
            return dict(self.values)

    def set_value(self, name: str, value: float) -> None:
        with self._lock:
            lo, hi = self.ranges[name]
            self.values[name] = min(max(float(value), lo), hi)
            if name.startswith("tendons.baseline.") and not (
                name.startswith("tendons.baseline.lengths.") or name.startswith("tendons.baseline.stiffness.")
            ):
                self._tendon_rebuild_requested = True

    def set_range(self, name: str, minimum: float | None = None, maximum: float | None = None) -> None:
        with self._lock:
            old_min, old_max = self.ranges[name]
            new_min = old_min if minimum is None else float(minimum)
            new_max = old_max if maximum is None else float(maximum)
            if new_min > new_max:
                new_min, new_max = new_max, new_min
            self.ranges[name] = (new_min, new_max)
            self.values[name] = min(max(self.values[name], new_min), new_max)

    def reset_value(self, spec: ParameterSpec) -> None:
        self.set_value(spec.name, spec.default)

    def consume_tendon_rebuild_request(self) -> bool:
        with self._lock:
            requested = self._tendon_rebuild_requested
            self._tendon_rebuild_requested = False
            return requested

    def set_controller(self, controller: str) -> None:
        with self._lock:
            self.controller = controller

    def get_controller(self) -> str:
        with self._lock:
            return self.controller

    def request_reset(self) -> None:
        with self._lock:
            self.reset_requested = True

    def consume_reset_request(self) -> bool:
        with self._lock:
            requested = self.reset_requested
            self.reset_requested = False
            return requested

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True
            self.paused = False
            self.reset_requested = True

    def should_stop(self) -> bool:
        with self._lock:
            return self.stop_requested

    def toggle_pause(self) -> bool:
        with self._lock:
            self.paused = not self.paused
            return self.paused

    def is_paused(self) -> bool:
        with self._lock:
            return self.paused

    def publish_telemetry(
        self,
        *,
        sim_time: float,
        controller_values: list[float],
        tendon_values: list[float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.telemetry = {"sim_time": float(sim_time), **(extra or {})}
            self.controller_history.append([float(value) for value in controller_values])
            self.tendon_history.append([float(value) for value in tendon_values])

    def latest_plot_values(self) -> tuple[list[float] | None, list[float] | None]:
        with self._lock:
            controller = self.controller_history[-1] if self.controller_history else None
            tendon = self.tendon_history[-1] if self.tendon_history else None
            return controller, tendon

    def latest_telemetry(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.telemetry)


def _range_around(value: float, *, radius: float, minimum: float | None = None) -> tuple[float, float]:
    lo = value - radius
    hi = value + radius
    if minimum is not None:
        lo = max(minimum, lo)
    if lo == hi:
        hi = lo + max(abs(value), 1.0)
    return lo, hi


def _spec(
    name: str,
    label: str,
    default: float,
    group: str,
    *,
    radius: float,
    minimum: float | None = None,
    unit: str = "",
):
    lo, hi = _range_around(float(default), radius=radius, minimum=minimum)
    step = max((hi - lo) / 200.0, 1.0e-6)
    return ParameterSpec(name, label, float(default), lo, hi, step, group, unit)


def _controller_specs(params) -> dict[str, list[ParameterSpec]]:
    cpg = params.run.cpg
    osc = params.run.cpg_oscillator
    sin = params.run.sinusoidal

    specs: dict[str, list[ParameterSpec]] = {
        "cpg": [
            _spec("run.cpg.f_hz", "Frequency", cpg.f_hz, "Timing", radius=2.0, minimum=0.01, unit="Hz"),
            _spec("run.cpg.duty_factor", "Duty factor", cpg.duty_factor, "Timing", radius=0.35, minimum=0.05),
            _spec(
                "run.cpg.hip_amplitude_deg",
                "Hip amplitude",
                cpg.hip_amplitude_deg,
                "Shape",
                radius=50.0,
                minimum=0.0,
                unit="deg",
            ),
            _spec("run.cpg.hip_offset_deg", "Hip offset", cpg.hip_offset_deg, "Shape", radius=50.0, unit="deg"),
            _spec(
                "run.cpg.knee_amplitude_deg",
                "Knee amplitude",
                cpg.knee_amplitude_deg,
                "Shape",
                radius=130.0,
                minimum=0.0,
                unit="deg",
            ),
            _spec("run.cpg.swing_start_offset", "Swing start", cpg.swing_start_offset, "Timing", radius=0.25),
            _spec("run.cpg.swing_end_offset", "Swing end", cpg.swing_end_offset, "Timing", radius=0.25),
            _spec(
                "run.cpg.combined_phase_offset_rad",
                "Combined phase",
                cpg.combined_phase_offset_rad,
                "Phase",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg.left_phase_offset_rad",
                "Left phase",
                cpg.left_phase_offset_rad,
                "Phase",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg.right_phase_offset_rad",
                "Right phase",
                cpg.right_phase_offset_rad,
                "Phase",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg.include_knee",
                "Include knee",
                1.0 if cpg.include_knee else 0.0,
                "Shape",
                radius=1.0,
                minimum=0.0,
            ),
        ],
        "cpg_oscillator": [
            _spec("run.cpg_oscillator.f_hz", "Frequency", osc.f_hz, "Timing", radius=2.0, minimum=0.01, unit="Hz"),
            _spec(
                "run.cpg_oscillator.duty_factor",
                "Duty factor",
                osc.duty_factor,
                "Timing",
                radius=0.35,
                minimum=0.05,
            ),
            _spec(
                "run.cpg_oscillator.left_phase_rad",
                "Left phase",
                osc.left_phase_rad,
                "Phase",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg_oscillator.right_phase_rad",
                "Right phase",
                osc.right_phase_rad,
                "Phase",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg_oscillator.hip_flexion_amplitude_deg",
                "Hip flex amp",
                osc.hip_flexion_amplitude_deg,
                "Hip flexion",
                radius=50.0,
                minimum=0.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.hip_flexion_offset_deg",
                "Hip flex offset",
                osc.hip_flexion_offset_deg,
                "Hip flexion",
                radius=50.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.hip_flexion_phase_rad",
                "Hip flex phase",
                osc.hip_flexion_phase_rad,
                "Hip flexion",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg_oscillator.knee_flexion_amplitude_deg",
                "Knee amp",
                osc.knee_flexion_amplitude_deg,
                "Knee",
                radius=80.0,
                minimum=0.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.knee_flexion_offset_deg",
                "Knee offset",
                osc.knee_flexion_offset_deg,
                "Knee",
                radius=80.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.knee_flexion_phase_rad",
                "Knee phase",
                osc.knee_flexion_phase_rad,
                "Knee",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg_oscillator.knee_swing_power",
                "Knee swing power",
                osc.knee_swing_power,
                "Knee",
                radius=2.5,
                minimum=0.1,
            ),
            _spec(
                "run.cpg_oscillator.hip_roll_amplitude_deg",
                "Hip roll amp",
                osc.hip_roll_amplitude_deg,
                "Hip roll",
                radius=30.0,
                minimum=0.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.hip_roll_offset_deg",
                "Hip roll offset",
                osc.hip_roll_offset_deg,
                "Hip roll",
                radius=30.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.hip_roll_phase_rad",
                "Hip roll phase",
                osc.hip_roll_phase_rad,
                "Hip roll",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.cpg_oscillator.hip_yaw_amplitude_deg",
                "Hip yaw amp",
                osc.hip_yaw_amplitude_deg,
                "Hip yaw",
                radius=30.0,
                minimum=0.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.hip_yaw_offset_deg",
                "Hip yaw offset",
                osc.hip_yaw_offset_deg,
                "Hip yaw",
                radius=30.0,
                unit="deg",
            ),
            _spec(
                "run.cpg_oscillator.hip_yaw_phase_rad",
                "Hip yaw phase",
                osc.hip_yaw_phase_rad,
                "Hip yaw",
                radius=6.283,
                unit="rad",
            ),
        ],
        "sin": [
            _spec("run.sinusoidal.f_hz", "Frequency", sin.f_hz, "Timing", radius=4.0, minimum=0.01, unit="Hz"),
            _spec(
                "run.sinusoidal.left_phi0_rad",
                "Left phi0",
                sin.left_phi0_rad,
                "Phase",
                radius=6.283,
                unit="rad",
            ),
            _spec(
                "run.sinusoidal.right_phi0_rad",
                "Right phi0",
                sin.right_phi0_rad,
                "Phase",
                radius=6.283,
                unit="rad",
            ),
        ],
    }

    for dof in DOF_ORDER:
        specs["sin"].append(
            _spec(
                f"run.sinusoidal.amplitude_deg.{dof}",
                f"{dof} amp",
                sin.amplitude_deg.get(dof, 0.0),
                "Amplitude",
                radius=90.0,
                minimum=0.0,
                unit="deg",
            )
        )
        specs["sin"].append(
            _spec(
                f"run.sinusoidal.offset_deg.{dof}",
                f"{dof} offset",
                sin.offset_deg.get(dof, 0.0),
                "Offset",
                radius=120.0,
                unit="deg",
            )
        )
        specs["sin"].append(
            _spec(
                f"run.sinusoidal.left_phase_rad.{dof}",
                f"left {dof} phase",
                sin.left_phase_rad.get(dof, 0.0),
                "Phase",
                radius=6.283,
                unit="rad",
            )
        )
        specs["sin"].append(
            _spec(
                f"run.sinusoidal.right_phase_rad.{dof}",
                f"right {dof} phase",
                sin.right_phase_rad.get(dof, 0.0),
                "Phase",
                radius=6.283,
                unit="rad",
            )
        )
    return specs


def _tendon_specs(params) -> list[ParameterSpec]:
    lengths = params.tendons.baseline.lengths
    stiffness = params.tendons.baseline.stiffness
    specs = [
        _spec(
            f"tendons.baseline.lengths.{name}",
            name,
            value,
            "Lengths",
            radius=max(abs(value) * 0.25, 0.02),
            minimum=0.0,
            unit="m",
        )
        for name, value in lengths.items()
    ]
    specs.extend(
        _spec(
            f"tendons.baseline.stiffness.{name}",
            f"{name} stiffness",
            value,
            "Stiffness",
            radius=max(abs(value), 1.0),
            minimum=0.0,
            unit="N/m",
        )
        for name, value in stiffness.items()
    )
    return specs


def _baseline_specs(params) -> list[ParameterSpec]:
    baseline = params.tendons.baseline
    specs: list[ParameterSpec] = []

    for name, value in baseline.joint_offsets_theta_deg.items():
        specs.append(
            _spec(
                f"tendons.baseline.joint_offsets_theta_deg.{name}",
                f"{name} theta offset",
                value,
                "Joint Offsets",
                radius=45.0,
                unit="deg",
            )
        )
    for name, value in baseline.joint_directions.items():
        specs.append(
            _spec(
                f"tendons.baseline.joint_directions.{name}",
                f"{name} direction",
                value,
                "Joint Directions",
                radius=2.0,
            )
        )
    for name, value in baseline.pulley_radii.items():
        specs.append(
            _spec(
                f"tendons.baseline.pulley_radii.{name}",
                f"{name} radius",
                value,
                "Pulley Radii",
                radius=max(abs(value) * 0.75, 0.01),
                minimum=0.0,
                unit="m",
            )
        )
    for name, value in baseline.chain_link_lengths.items():
        specs.append(
            _spec(
                f"tendons.baseline.chain_link_lengths.{name}",
                f"{name} length",
                value,
                "Chain Link Lengths",
                radius=max(abs(value) * 0.35, 0.03),
                minimum=0.0,
                unit="m",
            )
        )
    for name, value in baseline.connector_link_lengths_longitudinal.items():
        specs.append(
            _spec(
                f"tendons.baseline.connector_link_lengths_longitudinal.{name}",
                f"{name} long",
                value,
                "Connector Longitudinal",
                radius=max(abs(value) * 0.5, 0.02),
                minimum=0.0,
                unit="m",
            )
        )
    for name, value in baseline.connector_link_lengths_lateral.items():
        specs.append(
            _spec(
                f"tendons.baseline.connector_link_lengths_lateral.{name}",
                f"{name} lat",
                value,
                "Connector Lateral",
                radius=max(abs(value) * 1.0, 0.02),
                unit="m",
            )
        )
    for name, value in baseline.angles_deg.items():
        specs.append(
            _spec(
                f"tendons.baseline.angles_deg.{name}",
                name,
                value,
                "Angles",
                radius=45.0,
                unit="deg",
            )
        )
    return specs


def build_calibration_state(params, controller: str) -> CalibrationState:
    return CalibrationState(
        controller=controller,
        controller_specs=_controller_specs(params),
        tendon_specs=_tendon_specs(params),
        baseline_specs=_baseline_specs(params),
    )


_TENDON_ATTRS = {
    "tendons.baseline.lengths.gst_spring_rest": "gst_spring_rest_length",
    "tendons.baseline.lengths.upper_gst": "upper_gst_length",
    "tendons.baseline.lengths.lower_gst": "lower_gst_length",
    "tendons.baseline.lengths.dft": "dft_length",
    "tendons.baseline.lengths.edt1": "edt1_length",
    "tendons.baseline.lengths.edt2": "edt2_length",
    "tendons.baseline.lengths.kft": "kft_length",
    "tendons.baseline.stiffness.gst": "gst_stiffness",
    "tendons.baseline.stiffness.dft": "dft_stiffness",
    "tendons.baseline.stiffness.edt1": "edt1_stiffness",
    "tendons.baseline.stiffness.edt2": "edt2_stiffness",
    "tendons.baseline.stiffness.kft": "kft_stiffness",
}


def apply_tendon_parameters(tendon_data: Any, state: CalibrationState) -> None:
    """Apply hot tendon length/stiffness values to an existing ``TendonData`` object."""

    values = state.snapshot_values()
    for name, attr in _TENDON_ATTRS.items():
        if name not in values or not hasattr(tendon_data, attr):
            continue
        current = getattr(tendon_data, attr)
        if torch.is_tensor(current):
            current[:] = float(values[name])
        else:
            setattr(tendon_data, attr, float(values[name]))


def build_tendon_data_from_state(params, state: CalibrationState, *, num_instances: int, device: Any):
    """Create a fresh ``TendonData`` object from the current editable baseline values."""

    from isaaclab.tendons.models.analytic.tendon_data import TendonData

    live_params = copy.deepcopy(params)
    values = state.snapshot_values()
    prefix = "tendons.baseline."
    for name, value in values.items():
        if not name.startswith(prefix):
            continue
        remainder = name.removeprefix(prefix)
        section, key = remainder.rsplit(".", 1)
        target = getattr(live_params.tendons.baseline, section)
        target[key] = float(value)

    return TendonData(
        num_instances,
        live_params.to_tendon_randomization_ranges(),
        tc=live_params.to_tendon_constants(device=device),
        device=device,
    )
