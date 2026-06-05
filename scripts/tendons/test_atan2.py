# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

x = torch.tensor([-1.0])
y = torch.tensor([0.0])
x.requires_grad_()
y.requires_grad_()
atan2_result = torch.atan2(y, x)

# gradient check
atan2_result.sum().backward()
print("atan2 results:", atan2_result)
print("Gradients dx:", x.grad)
print("Gradients dy:", y.grad)
