# noise models for sensor data
# uses same random seed so all algos get idential noise

import numpy as np


class SensorNoiseModel:
    # adds noise to lidar and odometry readings
    # seeded rng ensures reproducability across different algorithm runs

    def __init__(self, config):
        self.lidar_noise_std = config['sensors']['lidar']['noise_std']
        self.odom_alphas = config['sensors']['odometry']['alphas']
        self.lidar_sparsity = config['sensors']['lidar'].get('sparsity', 1)

        seed = config.get('random_seed', 42)
        self.rng = np.random.RandomState(seed)  # seeded for fair comparison

    def add_lidar_noise(self, clean_ranges):
        # gaussian noise on lidar readings
        noise = self.rng.normal(0, self.lidar_noise_std, len(clean_ranges))
        noisy_ranges = clean_ranges + noise
        return np.clip(noisy_ranges, 0.12, 3.5)  # clip to sensor range

    def add_odometry_noise(self, v, w, dt):
        # motion model noise for odometry
        a1, a2, a3, a4 = self.odom_alphas

        d_trans = v * dt
        d_rot = w * dt

        # noise proportional to motion
        trans_std = a3 * abs(d_trans) + a4 * abs(d_rot)
        trans_noise = self.rng.normal(0, trans_std)

        rot_std = a1 * abs(d_rot) + a2 * abs(d_trans)
        rot_noise = self.rng.normal(0, rot_std)

        v_noisy = (d_trans + trans_noise) / dt if dt > 0 else v
        w_noisy = (d_rot + rot_noise) / dt if dt > 0 else w

        return v_noisy, w_noisy

    def apply_sparsity(self, ranges, angles=None):
        # use subset of lidar points (for sparsity test)
        if self.lidar_sparsity == 1:
            return (ranges, angles) if angles is not None else ranges

        indices = np.arange(0, len(ranges), self.lidar_sparsity)
        sparse_ranges = ranges[indices]

        if angles is not None:
            return sparse_ranges, angles[indices]
        return sparse_ranges

    def reset_seed(self, seed=None):
        # reset rng if needed
        self.rng = np.random.RandomState(seed if seed else 42)
