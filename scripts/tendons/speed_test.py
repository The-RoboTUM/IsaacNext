# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
import torch

# ----------------------------
#  PARAMETERS
# ----------------------------
N = 20000  # number of parallel worlds
dof = 2  # double pendulum
g = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
q_torch = torch.rand(N, dof, requires_grad=True, device="cuda")
grad_out = torch.ones(q_torch.shape[0], device=q_torch.device)

# ===============================================================
#  POTENTIAL ENERGY
# ===============================================================


def V_torch(q):
    q1 = q[..., 0]
    q2 = q[..., 1]
    return m1 * g * l1 * (1 - torch.cos(q1)) + m2 * g * (l1 * (1 - torch.cos(q1)) + l2 * (1 - torch.cos(q1 + q2)))


# ------- benchmark utility -------
def bench_torch_grad():
    V = V_torch(q_torch)
    (dVdq,) = torch.autograd.grad(V, q_torch, grad_outputs=grad_out)
    return dVdq


# ===============================================================
#  RUN BENCHMARK
# ===============================================================
def run(name, func):
    # warmup
    func()
    start = time.time()
    func()
    end = time.time()
    print(f"{name:30s}: {1000 * (end - start):.2f} ms")


print("\n=== PyTorch Benchmarks ===")
run("torch.autograd.grad", bench_torch_grad)
