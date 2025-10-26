# data I/O - load sensor data and save algorithm results

import numpy as np
import csv


def load_sensor_data(csv_path):
    # load sensor csv (timestamp, 360 lidar ranges, v, w, ground truth)
    timestamps = []
    lidar_scans = []
    odometry = []
    ground_truth = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))

            scan = np.array([float(row[f'range_{i}']) for i in range(360)])
            lidar_scans.append(scan)

            odometry.append([float(row['v']), float(row['w'])])

            ground_truth.append([float(row['gt_x']),
                                float(row['gt_y']),
                                float(row['gt_theta'])])

    return {
        'timestamps': np.array(timestamps),
        'lidar_scans': np.array(lidar_scans),
        'odometry': np.array(odometry),
        'ground_truth': np.array(ground_truth)
    }


def save_estimates(csv_path, timestamps, estimates):
    # save algorithm outputs to csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estimated_x', 'estimated_y', 'estimated_theta'])

        for t, est in zip(timestamps, estimates):
            writer.writerow([f"{t:.3f}", f"{est[0]:.6f}",
                           f"{est[1]:.6f}", f"{est[2]:.6f}"])


def save_timing(csv_path, timestamps, prediction_times, observation_times):
    # save timing info for performance analysis
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'prediction_time_ms', 'observation_time_ms'])

        for t, pred_t, obs_t in zip(timestamps, prediction_times, observation_times):
            writer.writerow([f"{t:.3f}", f"{pred_t:.3f}", f"{obs_t:.3f}"])


def compute_error_metrics(estimates, ground_truth):
    # calculate mae, rmse etc for comparison
    position_errors = np.sqrt(
        (estimates[:, 0] - ground_truth[:, 0])**2 +
        (estimates[:, 1] - ground_truth[:, 1])**2
    )

    return {
        'mae': np.mean(position_errors),
        'rmse': np.sqrt(np.mean(position_errors**2)),
        'max_error': np.max(position_errors),
        'position_errors': position_errors
    }
