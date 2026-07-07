# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live Isaac viewport overlay for Forrest tendon paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isaaclab.tendons.models.analytic.constants import link_names_left, link_names_right, tids
from isaaclab.tendons.models.analytic.visualization.context import DEFAULT_ALPHA_2, tc
from isaaclab.tendons.models.analytic.visualization.kinematics import compute_alphas, compute_joint_locations
from isaaclab.tendons.models.analytic.visualization.paths import (
    compute_dft_points,
    compute_edt1_points,
    compute_edt2_points,
    compute_gst_attachment_points,
    compute_kft_points,
)
from isaaclab.tendons.models.analytic.visualization.validation import arc_from_3_points


@dataclass(frozen=True)
class _LinePath:
    name: str
    points: list[list[float]]
    active_key: str
    active: bool | None = None


class ForrestTendonOverlay:
    """Draw calibration tendon paths with Isaac debug lines.

    This is debug overlay geometry, not authored USD geometry. In debug mode it
    reuses the same analytic tangent-point and pulley-arc construction used by
    ``scripts/tendons/draw_tendon_action.py``, then registers the 2-D analytic
    leg to the live 3-D robot links.
    """

    _ACTIVE_THICKNESS = 6.0
    _INACTIVE_THICKNESS = 2.0
    _PATH_OFFSETS = {
        "gst_upper": 0,
        "gst_lower": 1,
        "dft": 2,
        "kft": 3,
        "edt1": 4,
        "edt2": 5,
    }
    _BASE_COLORS = {
        "gst_upper": [1.00, 0.12, 0.06, 1.0],
        "gst_lower": [1.00, 0.35, 0.70, 1.0],
        "dft": [1.00, 0.58, 0.05, 1.0],
        "kft": [0.92, 0.82, 0.05, 1.0],
        "edt1": [0.10, 0.55, 1.00, 1.0],
        "edt2": [0.48, 0.22, 1.00, 1.0],
    }

    def __init__(self, robot: Any, *, update_interval: int = 3, line_thickness: float = 4.0):
        self.robot = robot
        self.update_interval = max(1, int(update_interval))
        self.line_thickness = float(line_thickness)
        self._draw = None
        self._warned = False
        self._body_indices = {
            "left": self._find_body_indices(link_names_left),
            "right": self._find_body_indices(link_names_right),
        }
        try:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._draw = omni_debug_draw.acquire_debug_draw_interface()
        except Exception as exc:  # pragma: no cover - depends on Isaac GUI extension availability.
            self._warn_once(f"Tendon overlay unavailable: {exc}")

    def clear(self) -> None:
        if self._draw is not None:
            self._draw.clear_lines()

    def update(
        self,
        *,
        iteration: int,
        left_debug: dict[str, Any] | None,
        right_debug: dict[str, Any] | None,
        tendon_data: Any | None = None,
        tendon_active: dict[str, bool] | None = None,
    ) -> None:
        """Redraw the current tendon overlay if the throttle interval elapsed."""

        if self._draw is None or iteration % self.update_interval != 0:
            return
        sources: list[list[float]] = []
        targets: list[list[float]] = []
        colors: list[list[float]] = []
        thicknesses: list[float] = []

        debug_by_side = {"left": left_debug, "right": right_debug}
        for side in ("left", "right"):
            debug_data = debug_by_side[side]
            if tendon_data is not None:
                self._sync_tendon_data(tendon_data)
            paths = self._analytic_paths(side, debug_data, tendon_active) if debug_data is not None else None
            if not paths:
                paths = self._body_paths(side, tendon_active)
            for path in paths:
                active = bool(
                    path.active if path.active is not None else (tendon_active or {}).get(path.active_key, True)
                )
                color = self._color_for(path.name, active)
                thickness = self._ACTIVE_THICKNESS if active else self._INACTIVE_THICKNESS
                for start, end in zip(path.points[:-1], path.points[1:]):
                    sources.append(start)
                    targets.append(end)
                    colors.append(color)
                    thicknesses.append(thickness)

        self._draw.clear_lines()
        if sources:
            self._draw.draw_lines(sources, targets, colors, thicknesses)

    def _sync_tendon_data(self, tendon_data: Any) -> None:
        """Refresh constants used by the reused analytic drawing helpers."""

        try:
            from isaaclab.tendons.models.analytic.visualization import context as viz_context

            for attr in (
                "pulley_radii",
                "link_lengths",
                "tendon_offsets_theta",
                "tendon_offsets_q_theta",
                "tendon_offsets_qhat_thetahat",
                "tendon_section_lengths",
                "tendon_tangency_angles",
            ):
                if hasattr(tendon_data, attr):
                    setattr(viz_context.td, attr, getattr(tendon_data, attr)[:1].detach().cpu())
            if hasattr(tendon_data, "pulley_radii"):
                viz_context.tc.pulley_radii = tendon_data.pulley_radii[0].detach().cpu()
        except Exception as exc:
            self._warn_once(f"Unable to sync tendon overlay constants: {exc}")

    def _find_body_indices(self, body_names: list[str]) -> list[int | None]:
        body_indices: list[int | None] = []
        for body_name in body_names:
            try:
                indices, _names = self.robot.find_bodies(body_name, preserve_order=True)
            except Exception:
                body_indices.append(None)
                continue
            if indices:
                body_indices.append(int(indices[0]))
            else:
                body_indices.append(None)
        return body_indices

    def _analytic_paths(
        self,
        side: str,
        data: dict[str, Any],
        tendon_active: dict[str, bool] | None,
    ) -> list[_LinePath] | None:
        try:
            thetas = np.asarray(data["thetas"], dtype=float)
            alphas = compute_alphas(DEFAULT_ALPHA_2, thetas)
            joints = compute_joint_locations(alphas)
            transform = self._fit_analytic_to_world(side, joints)
            if transform is None:
                return None

            (
                gst_upper_points,
                gst_upper_joints,
                _gst_upper_q_positives,
                gst_lower_points,
                gst_lower_joints,
                _gst_lower_q_positives,
            ) = compute_gst_attachment_points(DEFAULT_ALPHA_2, joints, data)
            dft_points, dft_joints, dft_q_positives = compute_dft_points(
                alphas,
                joints,
                data,
                tc.pulley_radii[tids.I_RADIUS_DFT_5].item(),
                tc.pulley_radii[tids.I_RADIUS_DFT_6].item(),
            )
            kft_points, kft_joints, kft_q_positives = compute_kft_points(
                alphas[5],
                joints,
                data,
                tc.pulley_radii[tids.I_RADIUS_KFT_8].item(),
            )
            edt1_points, edt1_joints, edt1_q_positives = compute_edt1_points(
                alphas,
                joints,
                data,
                tc.pulley_radii[tids.I_RADIUS_EDT1_5].item(),
            )
            edt2_points, edt2_joints, edt2_q_positives = compute_edt2_points(
                alphas,
                joints,
                data,
                tc.pulley_radii[tids.I_RADIUS_EDT2_5].item(),
                tc.pulley_radii[tids.I_RADIUS_EDT2_6].item(),
            )

            planar_paths = [
                (
                    "gst_upper",
                    self._tendon_path(gst_upper_points, gst_upper_joints, upper_tendon=True),
                    "gst",
                    self._debug_active(data, "GST", tendon_active, "gst"),
                ),
                (
                    "gst_lower",
                    self._tendon_path(gst_lower_points, gst_lower_joints, upper_tendon=False),
                    "gst",
                    self._debug_active(data, "GST", tendon_active, "gst"),
                ),
                (
                    "dft",
                    self._tendon_path_general(dft_points, dft_joints, dft_q_positives, [True, True]),
                    "dft",
                    self._debug_active(data, "DFT", tendon_active, "dft"),
                ),
                (
                    "kft",
                    self._tendon_path_general(kft_points, kft_joints, kft_q_positives, [False], start_with_arc=True),
                    "kft",
                    self._debug_active(data, "KFT", tendon_active, "kft"),
                ),
                (
                    "edt1",
                    self._tendon_path_general(edt1_points, edt1_joints, edt1_q_positives, [False]),
                    "edt1",
                    self._debug_active(data, "EDT1", tendon_active, "edt1"),
                ),
                (
                    "edt2",
                    self._tendon_path_general(edt2_points, edt2_joints, edt2_q_positives, [False, False]),
                    "edt2",
                    self._debug_active(data, "EDT2", tendon_active, "edt2"),
                ),
            ]

            return [
                _LinePath(name, self._planar_to_world(points, transform, side, name), active_key, active)
                for name, points, active_key, active in planar_paths
                if len(points) >= 2
            ]
        except Exception as exc:
            self._warn_once(f"Falling back to body-registered tendon overlay: {exc}")
            return None

    def _fit_analytic_to_world(
        self, side: str, joints: list[np.ndarray]
    ) -> tuple[np.ndarray, float, np.ndarray] | None:
        source_by_link = {
            tids.I_CHAIN_LINK_23: 0.5 * (joints[0] + joints[1]),
            tids.I_CHAIN_LINK_34: 0.5 * (joints[1] + joints[2]),
            tids.I_CHAIN_LINK_4prime5: 0.5 * (joints[2] + joints[3]),
            tids.I_CHAIN_LINK_56: 0.5 * (joints[3] + joints[4]),
            tids.I_CHAIN_LINK_67: 0.5 * (joints[4] + joints[5]),
        }
        src: list[np.ndarray] = []
        dst: list[np.ndarray] = []
        for link_index, source_point in source_by_link.items():
            body_index = self._body_index(side, link_index)
            if body_index is None:
                continue
            src.append(np.array([source_point[0], source_point[1], 0.0], dtype=float))
            dst.append(self.robot.data.body_link_pos_w[0, body_index].detach().cpu().numpy().astype(float))

        if len(src) < 3:
            return None

        source = np.stack(src, axis=0)
        target = np.stack(dst, axis=0)
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        source_centered = source - source_mean
        target_centered = target - target_mean
        covariance = source_centered.T @ target_centered / float(source.shape[0])
        u, singular_values, vt = np.linalg.svd(covariance)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
        variance = float((source_centered**2).sum() / source.shape[0])
        if variance <= 1.0e-12:
            return None
        scale = float(singular_values.sum() / variance)
        translation = target_mean - scale * (source_mean @ rotation.T)
        return rotation, scale, translation

    def _planar_to_world(
        self,
        points: list[np.ndarray],
        transform: tuple[np.ndarray, float, np.ndarray],
        side: str,
        name: str,
    ) -> list[list[float]]:
        rotation, scale, translation = transform
        source = np.array([[point[0], point[1], 0.0] for point in points], dtype=float)
        world = scale * (source @ rotation.T) + translation
        side_sign = 1.0 if side == "left" else -1.0
        offset_index = self._PATH_OFFSETS.get(name, 0)
        world[:, 1] += side_sign * (0.010 + 0.004 * offset_index)
        world[:, 2] += 0.010
        return world.tolist()

    def _body_index(self, side: str, link_index: int) -> int | None:
        indices = self._body_indices[side]
        if link_index >= len(indices):
            return None
        return indices[link_index]

    def _body_paths(self, side: str, tendon_active: dict[str, bool] | None = None) -> list[_LinePath]:
        indices = self._body_indices[side]
        side_sign = 1.0 if side == "left" else -1.0

        def point(link_index: int, offset_index: int) -> list[float] | None:
            if link_index >= len(indices) or indices[link_index] is None:
                return None
            position = self.robot.data.body_pos_w[0, indices[link_index]].detach().cpu().clone()
            position[1] += side_sign * (0.018 + 0.008 * offset_index)
            position[2] += 0.012
            return [float(position[0]), float(position[1]), float(position[2])]

        def path(name: str, link_ids: list[int], active_key: str, offset_index: int) -> _LinePath | None:
            points = [point(link_id, offset_index) for link_id in link_ids]
            points = [p for p in points if p is not None]
            if len(points) < 2:
                return None
            return _LinePath(name, points, active_key, bool((tendon_active or {}).get(active_key, True)))

        paths = [
            path("gst_upper", [tids.I_CHAIN_LINK_23, tids.I_CHAIN_LINK_34, tids.I_CHAIN_LINK_4prime5], "gst", 0),
            path("gst_lower", [tids.I_CHAIN_LINK_4prime5, tids.I_CHAIN_LINK_56, tids.I_CHAIN_LINK_67], "gst", 1),
            path("dft", [tids.I_CHAIN_LINK_4prime5, tids.I_CHAIN_LINK_56, tids.I_CHAIN_LINK_67], "dft", 2),
            path("kft", [tids.I_CHAIN_LINK_38, tids.I_CHAIN_LINK_23, tids.I_CHAIN_LINK_34], "kft", 3),
            path("edt1", [tids.I_CHAIN_LINK_34, tids.I_CHAIN_LINK_4prime5, tids.I_CHAIN_LINK_56], "edt1", 4),
            path(
                "edt2",
                [tids.I_CHAIN_LINK_34, tids.I_CHAIN_LINK_4prime5, tids.I_CHAIN_LINK_56, tids.I_CHAIN_LINK_67],
                "edt2",
                5,
            ),
        ]
        valid_paths = [candidate for candidate in paths if candidate is not None]
        if valid_paths:
            return valid_paths

        root = self.robot.data.root_pos_w[0].detach().cpu()
        hip = [float(root[0]), float(root[1] + side_sign * 0.18), float(root[2] - 0.35)]
        knee = [float(root[0] + 0.06), float(root[1] + side_sign * 0.18), float(root[2] - 0.70)]
        foot = [float(root[0] + 0.10), float(root[1] + side_sign * 0.18), float(root[2] - 1.00)]
        return [
            _LinePath("gst_upper", [hip, knee], "gst", bool((tendon_active or {}).get("gst", True))),
            _LinePath("gst_lower", [knee, foot], "gst", bool((tendon_active or {}).get("gst", True))),
        ]

    def _tendon_path(self, tendon_points: list, tendon_joints: list, *, upper_tendon: bool) -> list[np.ndarray]:
        tendon_points = list(tendon_points)
        tendon_joints = list(tendon_joints)
        arc = not upper_tendon
        last_point = tendon_points.pop(0)
        points = [np.asarray(last_point, dtype=float)]
        while tendon_points:
            current_point = np.asarray(tendon_points.pop(0), dtype=float)
            if not arc:
                points.append(current_point)
                arc = True
            else:
                current_joint = tendon_joints.pop(0)
                xs, ys = arc_from_3_points(
                    current_joint,
                    last_point,
                    current_point,
                    ccw=not (upper_tendon and len(tendon_joints) == 1),
                )
                points.extend(np.array([x, y], dtype=float) for x, y in zip(xs, ys))
                arc = False
            last_point = current_point
        return points

    def _tendon_path_general(
        self,
        tendon_points: list,
        tendon_joints: list,
        tendon_q_positives: list,
        joint_ccws: list,
        *,
        start_with_arc: bool = False,
    ) -> list[np.ndarray]:
        tendon_points = list(tendon_points)
        tendon_joints = list(tendon_joints)
        tendon_q_positives = list(tendon_q_positives)
        joint_ccws = list(joint_ccws)
        arc = start_with_arc
        last_point = np.asarray(tendon_points.pop(0), dtype=float)
        points = [last_point]
        while tendon_points:
            current_point = np.asarray(tendon_points.pop(0), dtype=float)
            if not arc:
                points.append(current_point)
                arc = True
            else:
                current_joint = tendon_joints.pop(0)
                current_q_positive = tendon_q_positives.pop(0)
                current_joint_ccw = joint_ccws.pop(0)
                xs, ys = arc_from_3_points(
                    current_joint,
                    last_point,
                    current_point,
                    ccw=current_joint_ccw,
                    q_positive=current_q_positive,
                )
                points.extend(np.array([x, y], dtype=float) for x, y in zip(xs, ys))
                arc = False
            last_point = current_point
        return points

    def _debug_active(
        self,
        data: dict[str, Any],
        debug_prefix: str,
        tendon_active: dict[str, bool] | None,
        active_key: str,
    ) -> bool:
        not_slack_key = f"{debug_prefix}_not_slack"
        if not_slack_key in data:
            return self._as_bool(data[not_slack_key])
        delta_key = f"{debug_prefix}_delta_L_s"
        if delta_key in data:
            return float(data[delta_key]) <= 0.0
        return bool((tendon_active or {}).get(active_key, True))

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, (list, tuple)):
            return any(self._as_bool(item) for item in value)
        return bool(value)

    def _color_for(self, name: str, active: bool) -> list[float]:
        color = list(self._BASE_COLORS.get(name, [1.0, 1.0, 1.0, 1.0]))
        if active:
            return color
        return [color[0] * 0.35, color[1] * 0.35, color[2] * 0.35, 0.35]

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        try:
            import carb

            carb.log_warn(message)
        except Exception:
            print(message)
