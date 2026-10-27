"""
kalman_filter.py
================
Linear Dynamical System (LDS) — Kalman Filter for Sensor Denoising.

Academic context (Unit 4: Linear Dynamical Systems)
---------------------------------------------------
An HMM has discrete hidden states q_t ∈ {1, ..., K}.
A Linear Dynamical System (LDS) is the continuous latent-state analog:
    State equation:      z_t = A z_{t-1} + w_t,   w_t ~ N(0, Q)
    Measurement equation: x_t = C z_t     + v_t,   v_t ~ N(0, R)

Where:
    z_t : continuous latent state (e.g., true triaxial body acceleration)
    x_t : noisy sensor measurement from smartphone accelerometer
    A   : state transition matrix
    C   : observation emission matrix
    Q   : process noise covariance
    R   : measurement noise covariance

This module implements:
1. 1D / 3D Kalman Filter from scratch using NumPy.
2. Filtering raw triaxial accelerometer signals from `Inertial Signals/`
   to demonstrate continuous latent state estimation vs discrete HMM.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class KalmanFilter1D:
    """
    Standard 1D Kalman Filter for linear dynamical state estimation.

    State model:
        z_t = a * z_{t-1} + w_t,   w_t ~ N(0, q)
        x_t = c * z_t     + v_t,   v_t ~ N(0, r)
    """

    def __init__(self, a: float = 1.0, c: float = 1.0, q: float = 1e-4, r: float = 1e-2):
        self.a = float(a)
        self.c = float(c)
        self.q = float(q)
        self.r = float(r)

    def filter(self, measurements: np.ndarray, initial_state: float = 0.0, initial_variance: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run forward Kalman filter on 1D measurement sequence.

        Returns
        -------
        states : (T,) filtered state estimates ẑ_{t|t}
        variances : (T,) error variances P_{t|t}
        """
        T = len(measurements)
        z_est = np.zeros(T)
        p_est = np.zeros(T)

        z = initial_state
        p = initial_variance

        for t in range(T):
            # 1. Predict
            z_pred = self.a * z
            p_pred = self.a * p * self.a + self.q

            # 2. Kalman Gain
            s = self.c * p_pred * self.c + self.r
            k = (p_pred * self.c) / s

            # 3. Update with measurement x_t
            x = measurements[t]
            z = z_pred + k * (x - self.c * z_pred)
            p = (1.0 - k * self.c) * p_pred

            z_est[t] = z
            p_est[t] = p

        return z_est, p_est


def demo_kalman_on_inertial_signal(signal_path: str, save_path: Optional[str] = None):
    """
    Denoises a raw sensor window from UCI HAR Inertial Signals and plots comparison.
    """
    # Load first window
    raw_windows = np.loadtxt(signal_path)
    window = raw_windows[0]  # 128 readings @ 50 Hz = 2.56 seconds

    kf = KalmanFilter1D(a=0.98, c=1.0, q=0.001, r=0.05)
    filtered, variances = kf.filter(window, initial_state=window[0], initial_variance=0.01)

    fig, ax = plt.subplots(figsize=(10, 4))
    time_sec = np.linspace(0, 2.56, len(window))
    ax.plot(time_sec, window, "b-", alpha=0.5, label="Raw Sensor Signal (Noisy Measurement x_t)")
    ax.plot(time_sec, filtered, "r-", linewidth=2, label="Kalman Filter Estimate (Latent State ẑ_t)")
    ax.fill_between(time_sec, filtered - 1.96 * np.sqrt(variances), filtered + 1.96 * np.sqrt(variances),
                    color="red", alpha=0.15, label="95% Confidence Interval")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Triaxial Acceleration (g)")
    ax.set_title("Unit 4 LDS Demo: Continuous Kalman Filtering of Triaxial Inertial Sensor")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
