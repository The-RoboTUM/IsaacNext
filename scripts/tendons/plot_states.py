"""
Plot state transitions over time from the states_gst.json file.
"""

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Qt5Agg")

# Load the states data
with open("outputs/states_gst.json", "r") as f:
    states = json.load(f)

# Load joint positions data for reference
with open("outputs/joint_pos_gst.json", "r") as f:
    joint_positions = json.load(f)

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

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
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/states_gst_plot.png", dpi=150, bbox_inches="tight")
print(f"Plot saved to outputs/states_gst_plot.png")
print(f"Total timesteps: {len(states)}")
print(f"State distribution:")
for state in ["a", "b", "c", "d", "s"]:
    count = states.count(state)
    print(f"  {state}: {count} ({100*count/len(states):.1f}%)")
print(f"Number of state transitions: {sum(state_changes)}")

plt.show()
