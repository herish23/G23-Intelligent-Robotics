# ================================================================
# File: ekf_sample.py
# Author: Liu Hao
# Version: v0.1
# ================================================================

import math


class EKF:
    """Basic structure for an Extended Kalman Filter (EKF)."""

    def __init__(self):
        # State vector [x, y, theta]
        self.x = [0.0, 0.0, 0.0]

        # Covariance matrix (3x3)
        self.P = [[0.0 for _ in range(3)] for _ in range(3)]

        # Process noise matrix (Q)
        self.Q = [[0.0 for _ in range(3)] for _ in range(3)]

        # Measurement noise matrix (R)
        self.R = [[0.0 for _ in range(2)] for _ in range(2)]

    # ------------------------------------------------------------
    def predict(self, control_input, dt=0.1):
        """
        EKF prediction step.
        Args:
            control_input: [v, w] (linear velocity, angular velocity)
            dt: time step
        """
        v, w = control_input
        x, y, theta = self.x

        # Motion model (simple differential drive)
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt

        self.x = [x, y, theta]

        # Simplified covariance update
        for i in range(3):
            for j in range(3):
                self.P[i][j] += self.Q[i][j]

    # ------------------------------------------------------------
    def update(self, measurement):
        """
        EKF update step.
        Args:
            measurement: [x_meas, y_meas]
        """
        # Measurement residual (difference)
        residual_x = measurement[0] - self.x[0]
        residual_y = measurement[1] - self.x[1]

        # Simplified Kalman gain (diagonal approximation)
        kx = 0.5  # Placeholder gain values
        ky = 0.5

        # Update state estimate
        self.x[0] += kx * residual_x
        self.x[1] += ky * residual_y

        # Simplified covariance update
        for i in range(2):
            self.P[i][i] *= (1 - 0.5)