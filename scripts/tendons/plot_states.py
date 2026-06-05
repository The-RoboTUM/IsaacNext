# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Plot state transitions over time from the states_gst.json file.
Compares left and right leg data side by side.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from isaaclab.tendons.legacy.constants_old import joint_names_left, joint_names_right

matplotlib.use("Qt5Agg")

# Load all data from JSONL files (left and right)
all_data_left = []
with open("outputs/gst_data_left.jsonl") as f:
    for line in f:
        all_data_left.append(json.loads(line))

all_data_right = []
with open("outputs/gst_data_right.jsonl") as f:
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

# Create figure with subplots (6 rows for each metric, 2 columns for left/right)
fig, axes = plt.subplots(6, 2, figsize=(16, 20))

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


def plot_states(ax, states, title, max_length):
    """Plot state transitions over time."""
    timesteps = np.arange(len(states))
    numeric_states = [state_to_numeric.get(state[0], -1) for state in states]
    ax.plot(
        timesteps,
        numeric_states,
        linewidth=1,
        marker="o",
        markersize=3,
        alpha=0.7,
    )
    # slack_indices = [i for i, state in enumerate(states) if state.startswith("s")]
    # ax.scatter(
    #     slack_indices,
    #     [numeric_states[i] for i in slack_indices],
    #     color="grey",
    #     s=30,
    #     label="Slack State",
    #     alpha=1.0,
    # )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("State")
    ax.set_title(title)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["a", "b", "c", "d", "s"])
    ax.set_xlim(0, max_length - 1)
    ax.grid(True, alpha=0.3)


def plot_state_changes(ax, states, title, max_length):
    """Plot state change events."""
    state_changes = [0] + [1 if states[i] != states[i - 1] else 0 for i in range(1, len(states))]
    change_indices = [i for i, change in enumerate(state_changes) if change == 1]
    ax.scatter(
        change_indices,
        [1] * len(change_indices),
        color="red",
        s=50,
        label="State Transitions",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("State Transition")
    ax.set_title(title)
    ax.set_ylim([0, 2])
    ax.set_xlim(0, max_length - 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return sum(state_changes)


def plot_lengths(ax, lengths, title, max_length):
    """Plot lengths over time."""
    timesteps_lengths = np.arange(len(lengths))
    ax.plot(timesteps_lengths, lengths, linewidth=1, color="green", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Length")
    ax.set_title(title)
    ax.set_xlim(0, max_length - 1)
    ax.grid(True, alpha=0.3)


def plot_joint_angles(ax, joint_positions, joint_names, title, max_length):
    """Plot joint angles over time."""
    joint_angles = [[x[i] for x in joint_positions] for i in range(len(joint_names))]
    for i in range(len(joint_names)):
        ax.plot(
            range(len(joint_angles[i])),
            joint_angles[i],
            label=joint_names[i],
            linewidth=1,
            alpha=0.7,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Joint Angle")
    ax.set_title(title)
    ax.set_xlim(0, max_length - 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_thetas(ax, thetas, joint_names, title, max_length):
    """Plot thetas over time."""
    thetas_values = [[x[i] for x in thetas] for i in range(len(joint_names))]
    for i in range(len(joint_names)):
        ax.plot(
            range(len(thetas_values[i])),
            thetas_values[i],
            label=joint_names[i],
            linewidth=1,
            alpha=0.7,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Theta")
    ax.set_title(title)
    ax.set_xlim(0, max_length - 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_torques(ax, torques, joint_names, title, max_length):
    """Plot torques over time for each joint."""
    torque_values = [[x[i] for x in torques] for i in range(len(joint_names))]
    for i in range(len(joint_names)):
        ax.plot(
            range(len(torque_values[i])),
            torque_values[i],
            label=joint_names[i],
            linewidth=1,
            alpha=0.7,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title(title)
    ax.set_xlim(0, max_length - 1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


# Row 0: States (Left and Right)
plot_states(axes[0, 0], states_left, "State Transitions Over Time - LEFT", max_length)
plot_states(axes[0, 1], states_right, "State Transitions Over Time - RIGHT", max_length)

# Row 1: State Changes (Left and Right)
n_transitions_left = plot_state_changes(axes[1, 0], states_left, "State Transition Events - LEFT", max_length)
n_transitions_right = plot_state_changes(axes[1, 1], states_right, "State Transition Events - RIGHT", max_length)

# Row 2: Lengths (Left and Right)
plot_lengths(axes[2, 0], lengths_left, "Lengths Over Time - LEFT", max_length)
plot_lengths(axes[2, 1], lengths_right, "Lengths Over Time - RIGHT", max_length)

# Row 3: Joint Angles (Left and Right)
plot_joint_angles(
    axes[3, 0],
    joint_positions_left,
    joint_names_left,
    "Joint Angles Over Time - LEFT",
    max_length,
)
plot_joint_angles(
    axes[3, 1],
    joint_positions_right,
    joint_names_right,
    "Joint Angles Over Time - RIGHT",
    max_length,
)

# Row 4: Thetas (Left and Right)
plot_thetas(axes[4, 0], thetas_left, joint_names_left, "Thetas Over Time - LEFT", max_length)
plot_thetas(axes[4, 1], thetas_right, joint_names_right, "Thetas Over Time - RIGHT", max_length)

# Row 5: Torques (Left and Right)
plot_torques(axes[5, 0], torques_left, joint_names_left, "Torques Over Time - LEFT", max_length)
plot_torques(
    axes[5, 1],
    torques_right,
    joint_names_right,
    "Torques Over Time - RIGHT",
    max_length,
)

plt.tight_layout()
plt.savefig("outputs/states_gst_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to outputs/states_gst_plot.png")

# Print statistics for both legs
for side, states in [("LEFT", states_left), ("RIGHT", states_right)]:
    print(f"\n=== {side} LEG ===")
    print(f"Total timesteps: {len(states)}")
    print("State distribution:")
    for state in ["a", "b", "c", "d", "s"]:
        count = states.count(state)
        print(f"  {state}: {count} ({100 * count / len(states):.1f}%)")

print(f"\nNumber of state transitions - LEFT: {n_transitions_left}")
print(f"Number of state transitions - RIGHT: {n_transitions_right}")

plt.show()
