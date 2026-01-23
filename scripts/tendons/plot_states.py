"""
Plot state transitions over time from the states_gst.json file.
"""

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from isaaclab.tendons.constants import (
    joint_names,
)

matplotlib.use("Qt5Agg")

# Load the states data
with open("outputs/states_gst.json", "r") as f:
    states = json.load(f)

# Load joint positions data for reference
with open("outputs/joint_pos_gst.json", "r") as f:
    joint_positions = json.load(f)

# Load lengths data
with open("outputs/lengths_gst.json", "r") as f:
    lengths = json.load(f)

# Load thetas data
with open("outputs/thetas_gst.json", "r") as f:
    thetas = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(5, 1, figsize=(12, 14))

# Determine the maximum x-axis length across all data
max_length = max(len(states), len(lengths), len(joint_positions), len(thetas))

# Plot 1: State over time
timesteps = np.arange(len(states))
state_to_numeric = {"a": 0, "b": 1, "c": 2, "d": 3, "s": 4}
numeric_states = [state_to_numeric.get(state, -1) for state in states]

ax1 = axes[0]
ax1.plot(timesteps, numeric_states, linewidth=1, marker="o", markersize=3, alpha=0.7)
ax1.set_xlabel("Timestep")
ax1.set_ylabel("State")
ax1.set_title("State Transitions Over Time (a, b, c, d, s)")
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.set_yticklabels(["a", "b", "c", "d", "s"])
ax1.set_xlim(0, max_length - 1)
ax1.grid(True, alpha=0.3)

# Plot 2: State changes (when transition occurs)
state_changes = [0] + [
    1 if states[i] != states[i - 1] else 0 for i in range(1, len(states))
]
ax2 = axes[1]
change_indices = [i for i, change in enumerate(state_changes) if change == 1]
ax2.scatter(
    change_indices,
    [1] * len(change_indices),
    color="red",
    s=50,
    label="State Transitions",
)
ax2.set_xlabel("Timestep")
ax2.set_ylabel("State Transition")
ax2.set_title("State Transition Events")
ax2.set_ylim([0, 2])
ax2.set_xlim(0, max_length - 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Lengths over time
ax3 = axes[2]
timesteps_lengths = np.arange(len(lengths))
ax3.plot(timesteps_lengths, lengths, linewidth=1, color="green", alpha=0.7)
ax3.set_xlabel("Timestep")
ax3.set_ylabel("Length")
ax3.set_title("Lengths Over Time")
ax3.set_xlim(0, max_length - 1)
ax3.grid(True, alpha=0.3)

# Plot 4: Joint angles over time
ax4 = axes[3]
joint_angles = [[x[i] for x in joint_positions] for i in range(len(joint_names))]
for i in range(len(joint_names)):
    ax4.plot(
        range(len(joint_angles[i])),
        joint_angles[i],
        label=joint_names[i],
        linewidth=1,
        alpha=0.7,
    )


ax4.set_xlabel("Timestep")
ax4.set_ylabel("Joint Angle")
ax4.set_title("Joint Angles Over Time")
ax4.set_xlim(0, max_length - 1)
ax4.legend(loc="best")
ax4.grid(True, alpha=0.3)

# Plot 5: Thetas over time
ax5 = axes[4]
thetas_values = [[x[i] for x in thetas] for i in range(len(joint_names))]
for i in range(len(joint_names)):
    ax5.plot(
        range(len(thetas_values[i])),
        thetas_values[i],
        label=joint_names[i],
        linewidth=1,
        alpha=0.7,
    )


ax5.set_xlabel("Timestep")
ax5.set_ylabel("Theta")
ax5.set_title("Thetas Over Time")
ax5.set_xlim(0, max_length - 1)
ax5.legend(loc="best")
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/states_gst_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to outputs/states_gst_plot.png")
print(f"Total timesteps: {len(states)}")
print("State distribution:")
for state in ["a", "b", "c", "d", "s"]:
    count = states.count(state)
    print(f"  {state}: {count} ({100 * count / len(states):.1f}%)")
print(f"Number of state transitions: {sum(state_changes)}")

plt.show()
