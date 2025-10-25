"""
Sensor noise models for LiDAR and odometry

IMPORTANT: All algorithms must use this to ensure identical noise!
"""

import numpy as np


class SensorNoiseModel:
    """
    Applies consistent sensor noise
    Uses seeded RNG for reproducibility
    """

    def __init__(self, config):
        self.lidar_noise_std = config['sensors']['lidar']['noise_std']
        self.odom_alphas = config['sensors']['odometry']['alphas']
        self.lidar_sparsity = config['sensors']['lidar'].get('sparsity', 1)

        # Seeded RNG for reproducibility
        seed = config.get('random_seed', 42)
        self.rng = np.random.RandomState(seed)

    def add_lidar_noise(self, clean_ranges):
        """Add Gaussian noise to LiDAR ranges"""
        noise = self.rng.normal(0, self.lidar_noise_std, len(clean_ranges))
        noisy_ranges = clean_ranges + noise
        # Clip to LDS-01 valid range
        return np.clip(noisy_ranges, 0.12, 3.5)

    def add_odometry_noise(self, v, w, dt):
        """
        Add motion model noise to odometry
        Based on Probabilistic Robotics Ch. 5
        """
        alpha1, alpha2, alpha3, alpha4 = self.odom_alphas

        # Motion magnitudes
        delta_trans = v * dt
        delta_rot = w * dt

        # Sample noise
        trans_noise_std = alpha3 * abs(delta_trans) + alpha4 * abs(delta_rot)
        trans_noise = self.rng.normal(0, trans_noise_std)

        rot_noise_std = alpha1 * abs(delta_rot) + alpha2 * abs(delta_trans)
        rot_noise = self.rng.normal(0, rot_noise_std)

        # Apply noise
        v_noisy = (delta_trans + trans_noise) / dt if dt > 0 else v
        w_noisy = (delta_rot + rot_noise) / dt if dt > 0 else w

        return v_noisy, w_noisy

    def apply_sparsity(self, ranges, angles=None):
        """Use only fraction of LiDAR points (for sparsity experiments)"""
        if self.lidar_sparsity == 1:
            return (ranges, angles) if angles is not None else ranges

        # Sample every Nth point
        indices = np.arange(0, len(ranges), self.lidar_sparsity)
        sparse_ranges = ranges[indices]

        if angles is not None:
            return sparse_ranges, angles[indices]
        return sparse_ranges

    def reset_seed(self, seed=None):
        """Reset RNG (useful for multiple runs)"""
        self.rng = np.random.RandomState(seed if seed else 42)
