# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draws the dynamic tendon state with matplotlib animation.

Controls:
    Space: Play/Pause
    Left/Right Arrow: Step frame backward/forward (when paused)
    Home/End: Jump to first/last frame
    d: Toggle debug geometry helper lines
    i: Toggle detailed debug info text

Usage:
    python scripts/tendons/draw_tendon_actuation.py
    python scripts/tendons/draw_tendon_actuation.py --save output.mp4
    python scripts/tendons/draw_tendon_actuation.py --record
    python scripts/tendons/draw_tendon_actuation.py --real-time
    python scripts/tendons/draw_tendon_actuation.py --show-debug-geometry --show-debug-text
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

DEFAULT_ALPHA_2_DEG = 280.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate tendon kinematics")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("outputs/gst_data_left.jsonl"),
        help=(
            "JSONL tendon debug data, forrest_tendons.db, forrest_kinematics.db, or a recording directory "
            "(default: outputs/gst_data_left.jsonl)."
        ),
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Side to load from database recordings (default: left).",
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="Save animation to MP4 file instead of displaying.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save a timestamped MP4 to outputs/ in the repository root.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for saved video (default: 30).",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Play according to wall-clock time and skip frames when rendering is slow.",
    )
    parser.add_argument(
        "--data-fps",
        type=float,
        default=20.0,
        help="Source data frame rate used by --real-time (default: 20).",
    )
    parser.add_argument(
        "--alpha-2-deg",
        type=float,
        default=DEFAULT_ALPHA_2_DEG,
        help=f"Base link angle in degrees (default: {DEFAULT_ALPHA_2_DEG:.1f}).",
    )
    parser.add_argument(
        "--parameters_file",
        type=str,
        default=None,
        help="Path to a Forrest parameter YAML file or profile directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timestamped progress messages.",
    )
    parser.add_argument(
        "--single-plot",
        action="store_true",
        help="Draw all tendons on one shared axis with separate active/inactive color pairs.",
    )
    parser.add_argument(
        "--show-debug-geometry",
        action="store_true",
        help="Show helper x/h geometry lines next to tendon paths.",
    )
    parser.add_argument(
        "--show-debug-text",
        action="store_true",
        help="Show detailed debug conditions in the side panels.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable per-frame geometry assertions while animating/saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.parameters_file:
        os.environ["ISAACNEXT_FORREST_CONFIG"] = args.parameters_file

    from isaaclab.tendons.models.analytic.visualization import (
        KinematicChainAnimator,
        TrajectoryOnlyAnimator,
        configure_plot_style,
        load_jsonl,
        load_recording,
    )
    from isaaclab.tendons.models.analytic.visualization.style import log

    configure_plot_style()

    if args.data.suffix == ".jsonl":
        all_data = load_jsonl(args.data)
        data_mode = "tendon"
    else:
        all_data, data_mode = load_recording(args.data, side=args.side)
    if not all_data:
        raise ValueError(f"No frames found in {args.data}")

    save_path = args.save
    if args.record:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(outputs_dir / f"draw_tendon_actuation_{timestamp}.mp4")

    if data_mode == "tendon":
        animator = KinematicChainAnimator(
            all_data,
            math.radians(args.alpha_2_deg),
            real_time=args.real_time and not save_path,
            data_fps=args.data_fps,
            verbose=args.verbose,
            single_plot=args.single_plot,
            show_debug_geometry=args.show_debug_geometry,
            show_debug_text=args.show_debug_text,
            validate_geometry=not args.no_validate,
        )
    else:
        animator = TrajectoryOnlyAnimator(
            all_data,
            math.radians(args.alpha_2_deg),
            real_time=args.real_time and not save_path,
            data_fps=args.data_fps,
            verbose=args.verbose,
        )

    if save_path:
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(fps=args.fps, metadata=dict(artist="IsaacLab"))
        log(f"Saving animation to {save_path}")
        animator.anim.save(save_path, writer=writer)
        log(f"Animation saved to {save_path}")
    else:
        animator.show()


if __name__ == "__main__":
    main()
