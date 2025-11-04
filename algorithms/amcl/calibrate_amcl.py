import sys
import numpy as np
import csv

sys.path.append("../../libraries")
from localization_utils import load_sensor_data, load_map, compute_likelihood_field

from amcl import initialize_particles, motion_update, sensor_update, normalize_weights, kld_resample, get_mean_pose
sensor_data = load_sensor_data('../../data/sensor_data_clean.csv')
map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
likelihood_field = compute_likelihood_field(map_info, max_dist=2.0)

print("Data loaded")
print(f"timesteps: {len(sensor_data['timestamps'])}")


## Baseline test 
## Params: alpha=0.2, sigma=0.2, beams=60, n_min=100, eps=0.05

def test_baseline():
    # current values in the amcl code
    particles = initialize_particles(100, map_info)
    n_steps = len(sensor_data['timestamps'])

    results = []

    print("\nRunning baseline test...")
    for t in range(1, n_steps):
        prev_odom = sensor_data['ground_truth'][t-1]
        curr_odom = sensor_data['ground_truth'][t]

        # motion update
        motion_update(particles, prev_odom, curr_odom)

        # sensor update
        lidar = sensor_data['lidar_scans'][t]
        sensor_update(particles, lidar, likelihood_field, map_info)
        normalize_weights(particles)

        # resample
        particles = kld_resample(particles, map_info)

        # estimate
        est_x, est_y, est_theta = get_mean_pose(particles)
        gt = sensor_data['ground_truth'][t]

        # error
        err = np.sqrt((est_x - gt[0])**2 + (est_y - gt[1])**2)

        results.append({
            't': t,
            'x_est': est_x,
            'y_est': est_y,
            'theta_est': est_theta,
            'x_gt': gt[0],
            'y_gt': gt[1],
            'theta_gt': gt[2],
            'error': err,
            'n_particles': len(particles)
        })

        if t % 50 == 0:
            print(f"t={t}: err={err:.3f}m, n={len(particles)}")

    # save to csv
    with open('results/baseline.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['t', 'x_est', 'y_est', 'theta_est','x_gt', 'y_gt', 'theta_gt','error', 'n_particles'])
        writer.writeheader()
        writer.writerows(results)

    # analysis
    errors = [r['error'] for r in results]
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)

    particles_count = [r['n_particles'] for r in results]
    mean_particles = np.mean(particles_count)

    print(f"Mean error: {mean_err:.3f}m")
    print(f"Std error: {std_err:.3f}m")
    print(f"Min error: {min_err:.3f}m")
    print(f"Max error: {max_err:.3f}m")
    print(f"Avg particles: {mean_particles:.1f}")

    return mean_err, std_err


# run baseline
test_baseline()
