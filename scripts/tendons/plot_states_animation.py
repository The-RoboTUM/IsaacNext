"""Create an animated MP4 of state transitions over time from the states_gst.json file.

Compares left and right leg data side by side with animation showing data developing.
"""

import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from isaaclab.tendons.constants_old import (
    joint_names_left,
    joint_names_right,
)

matplotlib.use("Agg")  # Use non-interactive backend for video generation

# Load all data from JSONL files (left and right)
all_data_left = []
with open("outputs/gst_data_left.jsonl", "r") as f:
    for line in f:
        all_data_left.append(json.loads(line))

all_data_right = []
with open("outputs/gst_data_right.jsonl", "r") as f:
    for line in f:
        all_data_right.append(json.loads(line))

# Extract individual data arrays from loaded data
states_left = [d["state"] for d in all_data_left]
states_right = [d["state"] for d in all_data_right]

joint_positions_left = [d["joint_pos"] for d in all_data_left]
joint_positions_right = [d["joint_pos"] for d in all_data_right]

lengths_left = [d["delta_l"] for d in all_data_left]
lengths_right = [d["delta_l"] for d in all_data_right]

thetas_left = [d["thetas"] for d in all_data_left]
thetas_right = [d["thetas"] for d in all_data_right]

torques_left = [d["tendon_torques"] for d in all_data_left]
torques_right = [d["tendon_torques"] for d in all_data_right]

# Determine the maximum x-axis length across all data
max_length = max(
    len(states_left),
    len(states_right),
    len(lengths_left),
    len(lengths_right),
    len(joint_positions_left),
    len(joint_positions_right),
    len(thetas_left),
    len(thetas_right),
    len(torques_left),
    len(torques_right),
)

state_to_numeric = {"a": 0, "b": 1, "c": 2, "d": 3, "s": 4}

# Pre-compute all data arrays
numeric_states_left = [state_to_numeric.get(state[0], -1) for state in states_left]
numeric_states_right = [state_to_numeric.get(state[0], -1) for state in states_right]

state_changes_left = [0] + [
    1 if states_left[i] != states_left[i - 1] else 0 for i in range(1, len(states_left))
]
state_changes_right = [0] + [
    1 if states_right[i] != states_right[i - 1] else 0
    for i in range(1, len(states_right))
]

joint_angles_left = [
    [x[i] for x in joint_positions_left] for i in range(len(joint_names_left))
]
joint_angles_right = [
    [x[i] for x in joint_positions_right] for i in range(len(joint_names_right))
]

thetas_values_left = [[x[i] for x in thetas_left] for i in range(len(joint_names_left))]
thetas_values_right = [
    [x[i] for x in thetas_right] for i in range(len(joint_names_right))
]

torque_values_left = [
    [x[i] for x in torques_left] for i in range(len(joint_names_left))
]
torque_values_right = [
    [x[i] for x in torques_right] for i in range(len(joint_names_right))
]

# Create figure with subplots (6 rows for each metric, 2 columns for left/right)
fig, axes = plt.subplots(6, 2, figsize=(16, 20))

# Setup all axes with proper limits and labels
for ax in axes.flat:
    ax.grid(True, alpha=0.3)


def set_margin(ax, min_val, max_val, margin_ratio=0.1):
    """Set y-axis limits with a margin."""
    margin = (max_val - min_val) * margin_ratio if max_val != min_val else 1
    ax.set_ylim(min_val - margin, max_val + margin)


def setup_axes():
    """Setup all axes with proper limits, labels, and empty lines."""
    lines = {}

    # Row 0: States
    for col, (ax, title) in enumerate(
        zip(
            axes[0],
            [
                "GST States - LEFT",
                "GST States - RIGHT",
            ],
        )
    ):
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("GST states")
        ax.text(
            0.5,
            0.92,
            title,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8
            ),
        )
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["a", "b", "c", "d", "s"])
        ax.set_xlim(0, max_length - 1)
        ax.set_ylim(-0.5, 4.5)
        (line,) = ax.plot([], [], linewidth=1, marker="o", markersize=3, alpha=0.7)
        lines[f"states_{col}"] = line

    # Row 1: State Changes
    for col, (ax, title) in enumerate(
        zip(
            axes[1],
            ["State Transition Events - LEFT", "State Transition Events - RIGHT"],
        )
    ):
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("State Transition")
        ax.text(
            0.5,
            0.92,
            title,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8
            ),
        )
        ax.set_ylim([0, 2])
        ax.set_xlim(0, max_length - 1)
        scatter = ax.scatter([], [], color="red", s=50, label="State Transitions")
        lines[f"changes_{col}"] = scatter
        ax.legend(loc="lower right")

    # Row 2: Lengths - compute shared y-limits
    all_lengths = lengths_left + lengths_right
    lengths_min, lengths_max = min(all_lengths), max(all_lengths)
    for col, (ax, title) in enumerate(
        zip(axes[2], ["GST Lengths - LEFT", "GST Lengths - RIGHT"])
    ):
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Length")
        ax.text(
            0.5,
            0.92,
            title,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8
            ),
        )
        ax.set_xlim(0, max_length - 1)
        if all_lengths:
            set_margin(ax, lengths_min, lengths_max)
        # Add black line at length=0 and shade area above grey
        ax.axhline(y=0, color="black", linewidth=1.5, zorder=5)
        ylim = ax.get_ylim()
        ax.fill_between(
            [0, max_length - 1], 0, ylim[1], color="grey", alpha=0.3, zorder=1
        )
        ax.set_ylim(ylim)  # Restore ylim after fill_between
        (line,) = ax.plot([], [], linewidth=1, color="green", alpha=0.7, zorder=10)
        lines[f"lengths_{col}"] = line

    # Row 3: Joint Angles - compute shared y-limits
    all_joint_values = [v for angles in joint_angles_left for v in angles] + [
        v for angles in joint_angles_right for v in angles
    ]
    joint_min, joint_max = min(all_joint_values), max(all_joint_values)
    for col, (ax, title, joint_names, joint_angles) in enumerate(
        zip(
            axes[3],
            ["Joint Angles - LEFT", "Joint Angles - RIGHT"],
            [joint_names_left, joint_names_right],
            [joint_angles_left, joint_angles_right],
        )
    ):
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Joint Angles")
        ax.text(
            0.5,
            0.92,
            title,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8
            ),
        )
        ax.set_xlim(0, max_length - 1)
        if all_joint_values:
            set_margin(ax, joint_min, joint_max)
        joint_lines = []
        for i, name in enumerate(joint_names):
            (line,) = ax.plot([], [], label=name, linewidth=1, alpha=0.7)
            joint_lines.append(line)
        lines[f"joints_{col}"] = joint_lines
        ax.legend(loc="lower right")

    # Row 4: Thetas - compute shared y-limits
    all_theta_values = [v for thetas in thetas_values_left for v in thetas] + [
        v for thetas in thetas_values_right for v in thetas
    ]
    theta_min, theta_max = min(all_theta_values), max(all_theta_values)
    for col, (ax, title, joint_names, thetas_vals) in enumerate(
        zip(
            axes[4],
            ["Thetas - LEFT", "Thetas - RIGHT"],
            [joint_names_left, joint_names_right],
            [thetas_values_left, thetas_values_right],
        )
    ):
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Theta")
        ax.text(
            0.5,
            0.92,
            title,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8
            ),
        )
        ax.set_xlim(0, max_length - 1)
        if all_theta_values:
            set_margin(ax, theta_min, theta_max)
        theta_lines = []
        for i, name in enumerate(joint_names):
            (line,) = ax.plot([], [], label=name, linewidth=1, alpha=0.7)
            theta_lines.append(line)
        lines[f"thetas_{col}"] = theta_lines
        ax.legend(loc="lower right")

    # Row 5: Torques - compute shared y-limits
    all_torque_values = [v for torques in torque_values_left for v in torques] + [
        v for torques in torque_values_right for v in torques
    ]
    torque_min, torque_max = min(all_torque_values), max(all_torque_values)
    for col, (ax, title, joint_names, torque_vals) in enumerate(
        zip(
            axes[5],
            ["Torques - LEFT", "Torques - RIGHT"],
            [joint_names_left, joint_names_right],
            [torque_values_left, torque_values_right],
        )
    ):
        ax.set_xlabel("Timestep")
        if col == 0:
            ax.set_ylabel("Torque [Nm]")
        ax.text(
            0.5,
            0.92,
            title,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8
            ),
        )
        ax.set_xlim(0, max_length - 1)
        if all_torque_values:
            set_margin(ax, torque_min, torque_max)
        torque_lines = []
        for i, name in enumerate(joint_names):
            (line,) = ax.plot([], [], label=name, linewidth=1, alpha=0.7)
            torque_lines.append(line)
        lines[f"torques_{col}"] = torque_lines
        ax.legend(loc="lower right")

    return lines


lines = setup_axes()
plt.tight_layout()


def update(frame):
    """Update function for animation."""
    # Calculate how many data points to show based on frame
    # We want to spread the animation over the total frames
    n_points = frame + 1

    # Row 0: States
    lines["states_0"].set_data(
        np.arange(min(n_points, len(numeric_states_left))),
        numeric_states_left[:n_points],
    )
    lines["states_1"].set_data(
        np.arange(min(n_points, len(numeric_states_right))),
        numeric_states_right[:n_points],
    )

    # Row 1: State Changes
    change_indices_left = [
        i
        for i in range(min(n_points, len(state_changes_left)))
        if state_changes_left[i] == 1
    ]
    change_indices_right = [
        i
        for i in range(min(n_points, len(state_changes_right)))
        if state_changes_right[i] == 1
    ]
    lines["changes_0"].set_offsets(
        np.column_stack([change_indices_left, [1] * len(change_indices_left)])
        if change_indices_left
        else np.empty((0, 2))
    )
    lines["changes_1"].set_offsets(
        np.column_stack([change_indices_right, [1] * len(change_indices_right)])
        if change_indices_right
        else np.empty((0, 2))
    )

    # Row 2: Lengths
    lines["lengths_0"].set_data(
        np.arange(min(n_points, len(lengths_left))), lengths_left[:n_points]
    )
    lines["lengths_1"].set_data(
        np.arange(min(n_points, len(lengths_right))), lengths_right[:n_points]
    )

    # Row 3: Joint Angles
    for i, line in enumerate(lines["joints_0"]):
        if i < len(joint_angles_left):
            line.set_data(
                np.arange(min(n_points, len(joint_angles_left[i]))),
                joint_angles_left[i][:n_points],
            )
    for i, line in enumerate(lines["joints_1"]):
        if i < len(joint_angles_right):
            line.set_data(
                np.arange(min(n_points, len(joint_angles_right[i]))),
                joint_angles_right[i][:n_points],
            )

    # Row 4: Thetas
    for i, line in enumerate(lines["thetas_0"]):
        if i < len(thetas_values_left):
            line.set_data(
                np.arange(min(n_points, len(thetas_values_left[i]))),
                thetas_values_left[i][:n_points],
            )
    for i, line in enumerate(lines["thetas_1"]):
        if i < len(thetas_values_right):
            line.set_data(
                np.arange(min(n_points, len(thetas_values_right[i]))),
                thetas_values_right[i][:n_points],
            )

    # Row 5: Torques
    for i, line in enumerate(lines["torques_0"]):
        if i < len(torque_values_left):
            line.set_data(
                np.arange(min(n_points, len(torque_values_left[i]))),
                torque_values_left[i][:n_points],
            )
    for i, line in enumerate(lines["torques_1"]):
        if i < len(torque_values_right):
            line.set_data(
                np.arange(min(n_points, len(torque_values_right[i]))),
                torque_values_right[i][:n_points],
            )

    print(f"Rendered frame {frame + 1}/{max_length}")
    return []


# Create animation
print(f"Creating animation with {max_length} frames at 30 fps...")
print(f"Estimated video duration: {max_length / 30:.1f} seconds")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=max_length,
    interval=1000 / 30,  # 30 fps
    blit=False,
)

# Save as MP4
output_path = "outputs/states_gst_animation.mp4"
print(f"Saving animation to {output_path}...")

writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="IsaacLab"), bitrate=2000)
ani.save(output_path, writer=writer)

print(f"Animation saved to {output_path}")

# Print statistics for both legs
for side, states in [("LEFT", states_left), ("RIGHT", states_right)]:
    print(f"\n=== {side} LEG ===")
    print(f"Total timesteps: {len(states)}")
    print("State distribution:")
    for state in ["a", "b", "c", "d", "s"]:
        count = states.count(state)
        print(f"  {state}: {count} ({100 * count / len(states):.1f}%)")
