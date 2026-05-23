"""CPG-based controller for BirdBot legs."""

from dataclasses import dataclass
import numpy as np

__all__ = ["BirdBotCPGLeg", "make_birdbot_cpg_controller"]

TWOPI = 2.0 * np.pi


def wrap_0_2pi(x):
    y = np.fmod(x, TWOPI)
    if y < 0:
        y += TWOPI
    return y


def theta_warp(phi, D):
    r"""Phase warping per duty factor D \in (0,1): map stance [0,2πD] -> [0,π], swing -> [π,2π]."""
    phi = wrap_0_2pi(phi)
    if phi <= TWOPI * D:
        return phi / (2.0 * D)  # linear map to [0,π]
    # second branch: Θ = φ/(2(1-D)) + 2π(1-2D)/(2(1-D))
    return phi / (2.0 * (1.0 - D)) + TWOPI * (1.0 - 2.0 * D) / (2.0 * (1.0 - D))


@dataclass
class CPGParams:
    f_hz: float = 1.0  # stride frequency [Hz]
    D: float = 0.6  # duty factor (stance fraction)
    A_h_deg: float = 32.0  # hip amplitude [deg] (gait 1: 32, gait 2: 35)
    O_h_deg: float = 22.0  # hip offset [deg] (gait 1: 22, gait 2: 30)
    A_k_deg: float = 120.0  # knee amplitude [deg]
    S_f: float = 0.0  # flexion delay fraction (0..1 of cycle)
    S_e: float = 0.22  # extension end fraction
    phi0: float = 0.0  # initial phase [rad]
    yaw_A_deg: float = (
        0.0  # optional yaw amplitude [deg] (default 0 → no yaw oscillation)
    )
    yaw_O_deg: float = 0.0  # yaw offset [deg]
    abd_A_deg: float = 0.0  # optional abduction amplitude [deg] (default 0)
    abd_O_deg: float = 0.0  # abduction offset [deg]


class BirdBotCPGLeg:

    def __init__(self, params: CPGParams, include_knee: bool = True):
        self.p = params
        self.include_knee = include_knee
        # Precompute radians
        self.A_h = np.deg2rad(self.p.A_h_deg)
        self.O_h = np.deg2rad(self.p.O_h_deg)
        self.A_k = np.deg2rad(self.p.A_k_deg)
        self.yaw_A = np.deg2rad(self.p.yaw_A_deg)
        self.yaw_O = np.deg2rad(self.p.yaw_O_deg)
        self.abd_A = np.deg2rad(self.p.abd_A_deg)
        self.abd_O = np.deg2rad(self.p.abd_O_deg)
        self.omega = TWOPI * self.p.f_hz

    def _phi(self, t):  # raw oscillator phase
        return self.omega * t + self.p.phi0

    def hip_flex(self, t):
        """Hip flexion angle and velocity (rad, rad/s)."""
        phi = self._phi(t)
        Th = theta_warp(phi, self.p.D)
        # piecewise constant dΘ/dt
        if wrap_0_2pi(phi) <= TWOPI * self.p.D:
            Thdot = self.omega / (2.0 * self.p.D)
        else:
            Thdot = self.omega / (2.0 * (1.0 - self.p.D))
        h = self.A_h * np.sin(Th + np.pi / 2) + self.O_h
        hdot = self.A_h * np.cos(Th + np.pi / 2) * Thdot
        return h, hdot

    def hip_abd(self, t):
        """Optional abduction pattern (simple sinus on raw phase)."""
        if self.abd_A == 0.0 and self.abd_O == 0.0:
            return 0.0, 0.0
        phi = self._phi(t)
        a = self.abd_A * np.sin(phi) + self.abd_O
        adot = self.abd_A * np.cos(phi) * self.omega
        return a, adot

    def hip_yaw(self, t):
        """Optional yaw pattern (simple sinus)."""
        if self.yaw_A == 0.0 and self.yaw_O == 0.0:
            return 0.0, 0.0
        phi = self._phi(t)
        y = self.yaw_A * np.sin(phi) + self.yaw_O
        ydot = self.yaw_A * np.cos(phi) * self.omega
        return y, ydot

    def knee(self, t):
        """Knee flexion command (rad, rad/s), gated amplitude (square-ish)."""
        if not self.include_knee:
            return 0.0, 0.0
        phi = wrap_0_2pi(self._phi(t))
        # Gate window: from 2πD+Sf to 2π - Se (converted from fractions to radians)
        phi_on_lo = TWOPI * self.p.D + TWOPI * self.p.S_f
        phi_on_hi = TWOPI - TWOPI * self.p.S_e
        amp = self.A_k if (phi >= phi_on_lo and phi <= phi_on_hi) else 0.0
        # Simple cosine knee with gated amplitude, in-phase with hip flex raw phase
        k = amp * np.maximum(0.0, np.cos(phi))  # non-negative flexion
        kdot = -amp * np.sin(phi) * self.omega if amp > 0 else 0.0
        return k, kdot

    # ----- Serial joint vector (yaw, abd, flex[, knee]) and velocities -----
    def q_serial(self, t):
        yaw, ydot = self.hip_yaw(t)
        abd, adot = self.hip_abd(t)
        flex, fdot = self.hip_flex(t)
        if self.include_knee:
            k, kdot = self.knee(t)
            return np.array([yaw, abd, flex, k]), np.array([ydot, adot, fdot, kdot])
        else:
            return np.array([yaw, abd, flex]), np.array([ydot, adot, fdot])


def make_birdbot_cpg_controller(params: CPGParams, map_matrix=None, include_knee=False):
    """Build a controller interface compatible with the actuator wrapper.

    Returns (qd_funs, qd_dot_funs) — lists of callables for each actuator.
    If map_matrix is None, outputs serial joint commands [yaw, abd, flex(, knee)].
    If map_matrix is provided (x = M*q), it maps serial joints into actuator space.

    Example M for differential (3x3):
        [[1, 0, 0],
         [0, 0.5, 0.5],
         [0, 0.5,-0.5]]
    """
    leg = BirdBotCPGLeg(params, include_knee=include_knee)

    # Evaluate serial joint functions
    def q_serial_fun(t):
        q, _ = leg.q_serial(t)
        return q

    def qd_serial_fun(t):
        _, qd = leg.q_serial(t)
        return qd

    # Build per-actuator functions
    if map_matrix is None:
        M = np.eye(4 if include_knee else 3)
    else:
        M = np.asarray(map_matrix, dtype=float)

    m = M.shape[0]

    def make_fun(i, vel=False):
        def f(t):
            q = q_serial_fun(t)
            if not vel:
                return float(M[i, : len(q)] @ q)
            else:
                qd = qd_serial_fun(t)
                return float(M[i, : len(qd)] @ qd)

        return f

    qd_funs = [make_fun(i, vel=False) for i in range(m)]
    qd_dot_funs = [make_fun(i, vel=True) for i in range(m)]
    return qd_funs, qd_dot_funs, leg
