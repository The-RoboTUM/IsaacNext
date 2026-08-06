# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime adapter for the deployed Identix Forrest elastic/tendon model."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch

from isaaclab.tendons.models.analytic.constants import joint_names_left, joint_names_right
from isaaclab.utils.math import euler_xyz_from_quat

FULL_ROBOT_NUM_DOFS = 30
BASE_COORDINATE_COUNT = 6
LEFT_LEG_OFFSET = 6
RIGHT_LEG_OFFSET = 18

REAL_LEG_JOINTS_LEFT = (
    "l0_acetabulofemoral_roll",
    "l1_acetabulofemoral_lateral",
    "lp1_pantograph",
    "l2_pseudo_acetabulofemoral_flexion",
    "l3b_femorotibial_back",
    "l3f_femorotibial_front",
    "l4f_intertarsal_front",
    "l4b_intertarsal_back",
    "l4p_intertarsal_pulley",
    "l5_metatarsophalangeal",
    "l6_interphalangeal",
    "l8_knee_flexor",
)
REAL_LEG_JOINTS_RIGHT = (
    "r0_acetabulofemoral_roll",
    "r1_acetabulofemoral_lateral",
    "rp1_pantograph",
    "r2_pseudo_acetabulofemoral_flexion",
    "r3b_femorotibial_back",
    "r3f_femorotibial_front",
    "r4f_intertarsal_front",
    "r4b_intertarsal_back",
    "r4p_intertarsal_pulley",
    "r5_metatarsophalangeal",
    "r6_interphalangeal",
    "r8_knee_flexor",
)

LEFT_TENDON_IDENTIX_DOFS = (11, 12, 15, 16, 17)
RIGHT_TENDON_IDENTIX_DOFS = (23, 24, 27, 28, 29)
DEFAULT_IDENTIX_BUNDLE_DIR = "../Identix/deploy/forrest_full_robot/20260806_140556"
DEFAULT_IDENTIX_REPO_PATH = "../Identix"
DEFAULT_JAX_LOG_LEVEL = "WARNING"


class IdentixForrestElasticTendonModel:
    """Call Identix ``elastic_force_batch(q)`` and expose Isaac tendon torques.

    The deployed Forrest model is trained in Identix's full robot coordinate
    order: q0..q5 are floating-base coordinates, q6..q17 are the left real-leg
    joints, and q18..q29 are the right real-leg joints. Its elastic-force output
    is the learned tendon generalized-force term in that same coordinate order.
    """

    def __init__(
        self,
        robot,
        *,
        bundle_dir: str | Path | None = None,
        identix_repo_path: str | Path | None = None,
        compile: bool = True,
        transfer: str = "auto",
        force_scale: float = 1.0,
        force_sign: float = 1.0,
        deployment: Any | None = None,
    ):
        self.robot = robot
        self.device = robot.device
        self.bundle_dir = _resolve_path(bundle_dir or DEFAULT_IDENTIX_BUNDLE_DIR)
        self.identix_repo_path = _resolve_path(identix_repo_path or DEFAULT_IDENTIX_REPO_PATH)
        self.compile = bool(compile)
        self.transfer = str(transfer)
        if self.transfer not in ("auto", "dlpack", "numpy"):
            raise ValueError("Identix transfer must be one of 'auto', 'dlpack', or 'numpy'.")
        self.force_scale = float(force_scale)
        self.force_sign = float(force_sign)

        self.left_real_leg_joint_indices = _resolve_joints(robot, REAL_LEG_JOINTS_LEFT)
        self.right_real_leg_joint_indices = _resolve_joints(robot, REAL_LEG_JOINTS_RIGHT)
        self.left_tendon_joint_indices = _resolve_joints(robot, tuple(joint_names_left))
        self.right_tendon_joint_indices = _resolve_joints(robot, tuple(joint_names_right))

        self._q_full: torch.Tensor | None = None
        self._deployment = deployment
        self._jax = None
        self.manifest = _read_manifest(self.bundle_dir)
        self._validate_manifest()

    def joint_torques(self, robot=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return left/right tendon-chain generalized forces for Isaac mapping."""

        robot = self.robot if robot is None else robot
        q_full = self.full_q_from_robot(robot)
        force_full = self.elastic_force(q_full)
        return self.tendon_torques_from_full_force(force_full)

    def full_q_from_robot(self, robot=None) -> torch.Tensor:
        """Pack current Isaac robot state into Identix full-robot q order."""

        robot = self.robot if robot is None else robot
        root_pos = robot.data.root_pos_w
        dtype = root_pos.dtype
        device = root_pos.device
        batch_size = int(root_pos.shape[0])
        if (
            self._q_full is None
            or self._q_full.shape != (batch_size, FULL_ROBOT_NUM_DOFS)
            or self._q_full.device != device
            or self._q_full.dtype != dtype
        ):
            self._q_full = torch.zeros((batch_size, FULL_ROBOT_NUM_DOFS), dtype=dtype, device=device)
        q_full = self._q_full
        q_full.zero_()

        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        q_full[:, 0:3] = root_pos
        q_full[:, 3:6] = torch.stack((roll, pitch, yaw), dim=-1).to(device=device, dtype=dtype)
        q_full[:, LEFT_LEG_OFFSET:RIGHT_LEG_OFFSET] = robot.data.joint_pos[:, self.left_real_leg_joint_indices]
        q_full[:, RIGHT_LEG_OFFSET:FULL_ROBOT_NUM_DOFS] = robot.data.joint_pos[:, self.right_real_leg_joint_indices]
        return q_full

    def elastic_force(self, q_full: torch.Tensor) -> torch.Tensor:
        """Run the deployed Identix model and return a Torch tensor on q's device."""

        q_full = q_full.detach().to(dtype=torch.float32).contiguous()
        if q_full.ndim != 2 or q_full.shape[1] != FULL_ROBOT_NUM_DOFS:
            raise ValueError(f"q_full must have shape (batch, {FULL_ROBOT_NUM_DOFS}), got {tuple(q_full.shape)}.")

        if self.transfer in ("auto", "dlpack"):
            try:
                return self._elastic_force_dlpack(q_full)
            except Exception as exc:
                if self.transfer == "dlpack":
                    raise RuntimeError("Identix DLPack transfer failed.") from exc

        return self._elastic_force_numpy(q_full)

    def tendon_torques_from_full_force(self, force_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Identix full-force output into current left/right tendon-chain torque tensors."""

        if force_full.ndim != 2 or force_full.shape[1] != FULL_ROBOT_NUM_DOFS:
            raise ValueError(
                f"Identix force output must have shape (batch, {FULL_ROBOT_NUM_DOFS}), got {tuple(force_full.shape)}."
            )
        force_full = torch.nan_to_num(force_full, nan=0.0, posinf=0.0, neginf=0.0)
        scale = self.force_scale * self.force_sign
        left = force_full.new_zeros((force_full.shape[0], len(LEFT_TENDON_IDENTIX_DOFS)))
        right = force_full.new_zeros((force_full.shape[0], len(RIGHT_TENDON_IDENTIX_DOFS)))
        left[:] = force_full[:, LEFT_TENDON_IDENTIX_DOFS] * scale
        right[:] = force_full[:, RIGHT_TENDON_IDENTIX_DOFS] * scale
        return left, right

    def _elastic_force_dlpack(self, q_full: torch.Tensor) -> torch.Tensor:
        deployment = self._load_deployment()
        jax = self._load_jax()
        from torch.utils import dlpack as torch_dlpack

        try:
            q_jax = jax.dlpack.from_dlpack(q_full)
        except TypeError:
            q_jax = jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(q_full))
        force_jax = deployment.elastic_force_batch(q_jax)
        try:
            force = torch_dlpack.from_dlpack(force_jax)
        except TypeError:
            force = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(force_jax))
        return force.to(device=q_full.device, dtype=q_full.dtype)

    def _elastic_force_numpy(self, q_full: torch.Tensor) -> torch.Tensor:
        deployment = self._load_deployment()
        import numpy as np

        force = np.asarray(deployment.elastic_force_batch(q_full.cpu().numpy()), dtype=np.float32)
        return torch.as_tensor(force, device=q_full.device, dtype=q_full.dtype)

    def _load_deployment(self):
        if self._deployment is not None:
            return self._deployment
        _configure_jax_logging()
        if self.identix_repo_path is not None:
            _prepend_python_path(self.identix_repo_path / "src")
            _prepend_python_path(self.identix_repo_path / "external" / "system_safari")
        try:
            from identix.deployment import load_deployment
        except Exception as exc:
            raise RuntimeError(
                "Could not import Identix deployment runtime. Install Identix and its dependencies in the Isaac "
                "Python environment, or set training.actions.tendon_model_type=analytic."
            ) from exc
        try:
            self._deployment = load_deployment(self.bundle_dir, compile=self.compile)
        except Exception as exc:
            raise RuntimeError(f"Could not load Identix deployment bundle: {self.bundle_dir}") from exc
        return self._deployment

    def _load_jax(self):
        if self._jax is not None:
            return self._jax
        _configure_jax_logging()
        try:
            import jax
        except Exception as exc:
            raise RuntimeError(
                "Could not import JAX for Identix DLPack transfer. Use transfer='numpy' for debugging or install JAX."
            ) from exc
        self._jax = jax
        return jax

    def _validate_manifest(self) -> None:
        if not self.manifest:
            return
        num_dofs = int(self.manifest.get("system", {}).get("num_dofs", FULL_ROBOT_NUM_DOFS))
        if num_dofs != FULL_ROBOT_NUM_DOFS:
            raise ValueError(f"Expected a {FULL_ROBOT_NUM_DOFS}-DOF Forrest Identix model, got {num_dofs}.")
        learned = tuple(self.manifest.get("components", {}).get("learned", ()))
        if learned and "elastic" not in learned:
            raise ValueError(f"Identix deployment does not include an elastic component: learned={learned}")


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _resolve_joints(robot, joint_names: tuple[str, ...]) -> list[int]:
    indices, found_names = robot.find_joints(list(joint_names), preserve_order=True)
    if tuple(found_names) != joint_names:
        raise RuntimeError(f"Could not resolve Forrest joints. Requested {joint_names}; found {tuple(found_names)}")
    return [int(index) for index in indices]


def _read_manifest(bundle_dir: Path | None) -> dict[str, Any]:
    if bundle_dir is None:
        return {}
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _prepend_python_path(path: Path) -> None:
    path_string = str(path)
    if path.exists() and path_string not in sys.path:
        sys.path.insert(0, path_string)


def _configure_jax_logging() -> None:
    os.environ.setdefault("JAX_LOG_COMPILES", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    level_name = os.environ.get("ISAACLAB_IDENTIX_JAX_LOG_LEVEL", DEFAULT_JAX_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.WARNING)
    for logger_name in ("jax", "jax._src"):
        logging.getLogger(logger_name).setLevel(level)
