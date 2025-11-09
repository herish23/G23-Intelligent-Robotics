## noise models for sensor data
## adds gaussian noise + outliers to lidar readings


import numpy as np


class SensorNoiseModel:

    def __init__(self, config):
        self.lidar_noise_std = config['sensors']['lidar']['noise_std']
        self.lidar_outlier_rate = config['sensors']['lidar'].get('outlier_rate', 0.0)
        self.odom_noise_std = config['sensors']['odometry'].get('noise_std', 0.01)
        self.lidar_max_range = config['sensors']['lidar'].get('max_range', 3.5)

        seed = config.get('random_seed', 42)
        self.rng = np.random.RandomState(seed)  ## seeded for fair comparison

    def add_lidar_noise(self, clean_ranges):
        ## gaussian noise on lidar readings
        noise = self.rng.normal(0, self.lidar_noise_std, len(clean_ranges))
        noisy_ranges = clean_ranges + noise

        if self.lidar_outlier_rate > 0:
            num_outliers = int(len(noisy_ranges) * self.lidar_outlier_rate)
            outlier_indices = self.rng.choice(len(noisy_ranges), num_outliers, replace=False)
            noisy_ranges[outlier_indices] = self.rng.uniform(0.12, self.lidar_max_range, num_outliers)

        return np.clip(noisy_ranges, 0.12, self.lidar_max_range)  ## clip to sensor range

    def add_odometry_noise(self, v, w, dt):
        ## gaussian noise on velocity commands
        v_noise = self.rng.normal(0, self.odom_noise_std)
        w_noise = self.rng.normal(0, self.odom_noise_std)

        v_noisy = v + v_noise
        w_noisy = w + w_noise

        return v_noisy, w_noisy
