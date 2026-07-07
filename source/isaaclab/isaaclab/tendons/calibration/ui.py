# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit UI windows for live Forrest tendon calibration."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from isaaclab.tendons.calibration.state import CalibrationState, ParameterSpec
from isaaclab.ui.widgets import LiveLinePlot

if TYPE_CHECKING:
    pass


class CalibrationWindows:
    """Owns the calibration controls window and the live plot window."""

    def __init__(self, state: CalibrationState, *, command_labels: list[str], tendon_labels: list[str]) -> None:
        import omni.kit.app
        import omni.ui as ui

        self._app = omni.kit.app.get_app()
        self._ui = ui
        self._state = state
        self._active_tab = "tendons"
        self._last_plot_update = 0.0
        self._plot_period = 1.0 / 25.0
        self._pause_button = None
        self._controller_buttons = {}
        self._tab_frame = None
        self._tendon_activity_lamps = {}

        self._control_window = ui.Window(
            "Forrest Calibration",
            width=760,
            height=900,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self._control_window.frame:
            self._build_controls()

        self._plot_window = ui.Window(
            "Forrest Calibration Plots",
            width=1280,
            height=420,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_BOTTOM,
        )
        with self._plot_window.frame:
            with ui.HStack(spacing=8):
                with ui.VStack(spacing=4):
                    ui.Label("Controller Commands", height=20)
                    self._command_plot = LiveLinePlot(
                        [[] for _ in command_labels],
                        y_min=-1.5,
                        y_max=1.5,
                        plot_height=250,
                        legends=command_labels,
                        max_datapoints=600,
                    )
                with ui.VStack(spacing=4):
                    ui.Label("Tendon Torques", height=20)
                    self._tendon_plot = LiveLinePlot(
                        [[] for _ in tendon_labels],
                        y_min=-1.0,
                        y_max=1.0,
                        plot_height=250,
                        legends=tendon_labels,
                        max_datapoints=600,
                    )
                    self._build_tendon_activity_strip()

        asyncio.ensure_future(self._dock_windows())

    def destroy(self) -> None:
        if self._control_window is not None:
            self._control_window.visible = False
            self._control_window.destroy()
            self._control_window = None
        if self._plot_window is not None:
            self._plot_window.visible = False
            self._plot_window.destroy()
            self._plot_window = None

    def update(self) -> None:
        now = time.perf_counter()
        if now - self._last_plot_update < self._plot_period:
            return
        self._last_plot_update = now
        controller_values, tendon_values = self._state.latest_plot_values()
        if controller_values is not None:
            self._command_plot.add_datapoint(controller_values)
        if tendon_values is not None:
            self._tendon_plot.add_datapoint(tendon_values)
        if self._pause_button is not None:
            self._pause_button.text = "Resume" if self._state.is_paused() else "Pause"
        self._update_tendon_activity()

    def _build_controls(self) -> None:
        ui = self._ui
        with ui.VStack(spacing=3):
            with ui.HStack(height=28, spacing=2):
                ui.Label("Controller", width=82)
                for controller, label in (("cpg", "CPG"), ("cpg_oscillator", "Oscillator"), ("sin", "Sine")):
                    self._controller_buttons[controller] = ui.Button(
                        label,
                        width=78,
                        clicked_fn=lambda c=controller: self._select_controller(c),
                    )

            with ui.HStack(height=28, spacing=2):
                ui.Button("Tendons / Model", width=132, clicked_fn=lambda: self._select_tab("tendons"))
                ui.Button("Baseline Geometry", width=144, clicked_fn=lambda: self._select_tab("baseline"))
                ui.Button("Controller Params", width=140, clicked_fn=lambda: self._select_tab("controller"))

            self._tab_frame = ui.Frame(height=ui.Fraction(1))
            self._rebuild_tab()

            with ui.HStack(height=30, spacing=2):
                ui.Button("Reset sim", width=96, clicked_fn=self._state.request_reset)
                self._pause_button = ui.Button("Pause", width=72, clicked_fn=self._toggle_pause)
                ui.Button("Reset and stop", width=116, clicked_fn=self._state.request_stop)

    def _toggle_pause(self) -> None:
        self._state.toggle_pause()
        if self._pause_button is not None:
            self._pause_button.text = "Resume" if self._state.is_paused() else "Pause"

    def _select_controller(self, controller: str) -> None:
        self._state.set_controller(controller)
        if self._active_tab == "controller":
            self._rebuild_tab()

    def _select_tab(self, tab: str) -> None:
        self._active_tab = tab
        self._rebuild_tab()

    def _rebuild_tab(self) -> None:
        if self._tab_frame is None:
            return
        if self._active_tab == "controller":
            self._tab_frame.set_build_fn(self._build_controller_tab)
        elif self._active_tab == "baseline":
            self._tab_frame.set_build_fn(self._build_baseline_tab)
        else:
            self._tab_frame.set_build_fn(self._build_tendon_tab)
        self._tab_frame.rebuild()

    def _build_tendon_tab(self) -> None:
        ui = self._ui
        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with ui.VStack(spacing=5):
                self._build_parameter_header()
                for spec in self._state.tendon_specs:
                    self._build_parameter_row(spec)

    def _build_baseline_tab(self) -> None:
        ui = self._ui
        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with ui.VStack(spacing=5):
                self._build_parameter_header()
                last_group = None
                for spec in self._state.baseline_specs:
                    if spec.group != last_group:
                        ui.Label(spec.group, height=20)
                        last_group = spec.group
                    self._build_parameter_row(spec)

    def _build_controller_tab(self) -> None:
        ui = self._ui
        controller = self._state.get_controller()
        specs = self._state.controller_specs[controller]
        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with ui.VStack(spacing=5):
                ui.Label(controller, height=20)
                self._build_parameter_header()
                for spec in specs:
                    self._build_parameter_row(spec)

    def _build_parameter_header(self) -> None:
        ui = self._ui
        with ui.HStack(height=20, spacing=4):
            ui.Label("Parameter", width=150)
            ui.Label("Slider", width=220)
            ui.Label("Value", width=80)
            ui.Label("Min", width=70)
            ui.Label("Max", width=70)

    def _build_parameter_row(self, spec: ParameterSpec) -> None:
        ui = self._ui
        value = self._state.values[spec.name]
        lo, hi = self._state.ranges[spec.name]
        with ui.HStack(height=34, spacing=4):
            ui.Label(spec.label, width=150)
            slider = ui.FloatSlider(min=lo, max=hi, step=spec.step, width=220)
            slider.model.set_value(value)
            value_drag = ui.FloatDrag(width=80, min=lo, max=hi, step=spec.step)
            value_drag.model.set_value(value)
            min_drag = ui.FloatDrag(width=70, step=spec.step)
            min_drag.model.set_value(lo)
            max_drag = ui.FloatDrag(width=70, step=spec.step)
            max_drag.model.set_value(hi)
            ui.Button("Reset", width=54, clicked_fn=lambda s=spec: self._reset_parameter(s))

            updating = False

            def _set_value(model, name=spec.name):
                nonlocal updating
                if updating:
                    return
                updating = True
                try:
                    new_value = model.as_float
                    self._state.set_value(name, new_value)
                    slider.model.set_value(self._state.values[name])
                    value_drag.model.set_value(self._state.values[name])
                finally:
                    updating = False

            def _set_min(model, name=spec.name):
                nonlocal updating
                if updating:
                    return
                updating = True
                try:
                    self._state.set_range(name, minimum=model.as_float)
                    new_min, new_max = self._state.ranges[name]
                    slider.min = new_min
                    value_drag.min = new_min
                    max_drag.model.set_value(new_max)
                finally:
                    updating = False

            def _set_max(model, name=spec.name):
                nonlocal updating
                if updating:
                    return
                updating = True
                try:
                    self._state.set_range(name, maximum=model.as_float)
                    new_min, new_max = self._state.ranges[name]
                    slider.max = new_max
                    value_drag.max = new_max
                    min_drag.model.set_value(new_min)
                finally:
                    updating = False

            slider.model.add_value_changed_fn(_set_value)
            value_drag.model.add_value_changed_fn(_set_value)
            min_drag.model.add_value_changed_fn(_set_min)
            max_drag.model.add_value_changed_fn(_set_max)

    def _reset_parameter(self, spec: ParameterSpec) -> None:
        self._state.reset_value(spec)
        self._rebuild_tab()

    def _build_tendon_activity_strip(self) -> None:
        ui = self._ui
        names = ("GST", "DFT", "KFT", "EDT1", "EDT2")
        with ui.HStack(height=28, spacing=6):
            for name in names:
                with ui.HStack(width=68, spacing=3):
                    lamp = ui.Rectangle(width=12, height=12, style={"background_color": 0xFF383838})
                    ui.Label(name, width=44)
                    self._tendon_activity_lamps[name] = lamp

    def _update_tendon_activity(self) -> None:
        active = self._state.latest_telemetry().get("tendon_active", {})
        for name, lamp in self._tendon_activity_lamps.items():
            is_active = bool(active.get(name.lower(), False))
            lamp.style = {"background_color": 0xFF33CC66 if is_active else 0xFF383838}

    async def _dock_windows(self) -> None:
        ui = self._ui
        for _ in range(20):
            if ui.Workspace.get_window("Forrest Calibration") and ui.Workspace.get_window("Forrest Calibration Plots"):
                break
            await self._app.next_update_async()

        control = ui.Workspace.get_window("Forrest Calibration")
        plots = ui.Workspace.get_window("Forrest Calibration Plots")
        right_target = self._first_window(("Stage", "Property", "Properties"))
        content_target = self._first_window(("Content", "Content Browser"))

        self._hide_windows(("Property", "Properties", "Semantics Schema Editor"))

        if control is not None and right_target is not None:
            control.visible = True
            control.width = 760
            control.height = 900
            control.dock_in(right_target, ui.DockPosition.SAME, 1.0)
            control.focus()
        if plots is not None and content_target is not None:
            plots.visible = True
            plots.width = 1280
            plots.height = 420
            plots.dock_in(content_target, ui.DockPosition.SAME, 1.0)
            plots.focus()

    def _first_window(self, names: tuple[str, ...]):
        for name in names:
            window = self._ui.Workspace.get_window(name)
            if window is not None:
                return window
        return None

    def _hide_windows(self, names: tuple[str, ...]) -> None:
        for name in names:
            window = self._ui.Workspace.get_window(name)
            if window is not None:
                window.visible = False
