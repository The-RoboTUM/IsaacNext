# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State extraction helpers for tendon debug frames."""

from __future__ import annotations


def get_gst_state(data):
    state_a = data["GST_state_a"]
    state_b = data["GST_state_b"]
    state_c = data["GST_state_c"]
    state_d = data["GST_state_d"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    elif state_c:
        return "c"
    elif state_d:
        return "d"
    else:
        raise ValueError("GST: no state is true")


def get_dft_state(data):
    state_a = data["DFT_state_A"]
    state_b = data["DFT_state_B"]
    state_c = data["DFT_state_C"]
    state_d = data["DFT_state_D"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    elif state_c:
        return "c"
    elif state_d:
        return "d"
    else:
        raise ValueError("DFT: no state is true")


def get_edt1_state(data):
    state_a = data["EDT1_state_a"]
    state_b = data["EDT1_state_b"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    else:
        raise ValueError("EDT1: no state is true")


def get_edt2_state(data):
    state_a = data["EDT2_state_a"]
    state_b = data["EDT2_state_b"]
    state_c = data["EDT2_state_c"]
    state_d = data["EDT2_state_d"]
    if state_a:
        return "a"
    elif state_b:
        return "b"
    elif state_c:
        return "c"
    elif state_d:
        return "d"
    else:
        raise ValueError("EDT2: no state is true")
