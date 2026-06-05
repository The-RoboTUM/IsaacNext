# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")

with open("outputs/joint_pos_passive.json") as f:
    passive = json.load(f)

with open("outputs/joint_pos_baseline.json") as f:
    baseline = json.load(f)

with open("outputs/joint_pos_v2_no_act.json") as f:
    v2_no_act = json.load(f)

with open("outputs/joint_pos_v2_no_act_j2_inv.json") as f:
    v2_no_act_j2_inv = json.load(f)

passive_j1 = [x[0] for x in passive]
passive_j2 = [x[1] for x in passive]

baseline_j1 = [x[0] for x in baseline]
baseline_j2 = [x[1] for x in baseline]

v2_no_act_j1 = [x[0] for x in v2_no_act]
v2_no_act_j2 = [x[1] for x in v2_no_act]

v2_no_act_j2_inv_j1 = [x[0] for x in v2_no_act_j2_inv]
v2_no_act_j2_inv_j2 = [x[1] for x in v2_no_act_j2_inv]

fig = plt.figure(figsize=(5, 10))
axs = fig.subplots(2, 1)


axs[0].plot(list(range(len(passive_j1))), passive_j1, color="blue")
axs[0].plot(list(range(len(baseline_j1))), baseline_j1, color="green")
axs[0].plot(list(range(len(v2_no_act_j1))), v2_no_act_j1, color="orange")
axs[0].plot(list(range(len(v2_no_act_j2_inv_j1))), v2_no_act_j2_inv_j1, color="red")
axs[1].plot(list(range(len(passive_j2))), passive_j2, color="blue")
axs[1].plot(list(range(len(baseline_j2))), baseline_j2, color="green")
axs[1].plot(list(range(len(v2_no_act_j2))), v2_no_act_j2, color="orange")
axs[1].plot(list(range(len(v2_no_act_j2_inv_j2))), v2_no_act_j2_inv_j2, color="red")


plt.show()
