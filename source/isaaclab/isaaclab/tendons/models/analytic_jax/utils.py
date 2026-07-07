# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for tendons."""

import jax.numpy as jnp



def list_from_dict(d: dict, n: int) -> list:
    """Convert a dict of lists to a list of lists."""
    assert min(d.keys()) == 0 and max(d.keys()) == n - 1 and len(set(d.keys())) == n, (
        "Dict keys must be consecutive integers starting from 0."
    )
    return [d[k] for k in sorted(d.keys())]


def dbg_grad(name: str, x: jnp.ndarray, joint_angles: jnp.ndarray):
    print(f"\n--- {name} ---")
    print("shape:", tuple(x.shape))
    print("requires_grad:", x.requires_grad)
    print("grad_fn:", x.grad_fn)
    print("is_leaf:", x.is_leaf)

    if x.requires_grad:
        try:
            g = jnp.autograd.grad(
                outputs=x.sum(),
                inputs=joint_angles,
                retain_graph=True,
                allow_unused=True,
            )[0]
            print("connected_to_joint_angles:", g is not None)
            if g is not None:
                print("grad norm:", g.norm().item())
                print("grad has nan:", jnp.isnan(g).any().item())
        except Exception as e:
            print("grad test failed:", repr(e))
    else:
        print("connected_to_joint_angles: False")
