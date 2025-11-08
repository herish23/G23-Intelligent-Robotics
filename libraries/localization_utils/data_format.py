## data I/O - load sensor csv and save results

import numpy as np
import csv


def load_sensor_data(csv_path):
    ## load sensor csv (timestamp, 360 lidar, odometry from encoders, ground truth)
    timestamps = []
    lidar_scans = []
    odometry = []
    ground_truth = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))

            ## 360 lidar beams
            scan = np.array([float(row[f'range_{i}']) for i in range(360)])
            lidar_scans.append(scan)

            ## odometry from wheel encoders (x, y, theta)
            odom = [float(row['odom_x']),
                    float(row['odom_y']),
                    float(row['odom_theta'])]
            odometry.append(odom)

            ## ground truth for error calc
            gt = [float(row['gt_x']),
                  float(row['gt_y']),
                  float(row['gt_theta'])]
            ground_truth.append(gt)

    return {
        'timestamps': np.array(timestamps),
        'lidar_scans': np.array(lidar_scans),
        'odometry': np.array(odometry),
        'ground_truth': np.array(ground_truth)
    }


def save_estimates(csv_path, timestamps, estimates):
    ## save algorithm results to csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'estimated_x', 'estimated_y', 'estimated_theta'])

        for t, est in zip(timestamps, estimates):
            writer.writerow([f"{t:.3f}", f"{est[0]:.6f}",
                           f"{est[1]:.6f}", f"{est[2]:.6f}"])


def save_timing(csv_path, timestamps, pred_times, obs_times):
    ## save timing for performance analysis
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'prediction_time_ms', 'observation_time_ms'])

        for t, pt, ot in zip(timestamps, pred_times, obs_times):
            writer.writerow([f"{t:.3f}", f"{pt:.3f}", f"{ot:.3f}"])


def compute_error_metrics(estimates, ground_truth):
    ## calc mae, rmse for comparing algorithms
    pos_err = np.sqrt(
        (estimates[:, 0] - ground_truth[:, 0])**2 +
        (estimates[:, 1] - ground_truth[:, 1])**2
    )

    return {
        'mae': np.mean(pos_err),
        'rmse': np.sqrt(np.mean(pos_err**2)),
        'max_error': np.max(pos_err),
        'position_errors': pos_err
    }
