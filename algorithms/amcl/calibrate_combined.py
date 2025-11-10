import sys
import numpy as np
import csv
import json
import time
from datetime import datetime

sys.path.append("../../libraries")
from localization_utils import load_sensor_data, load_map, compute_likelihood_field

## import from amcl but override all params
import amcl
from amcl import initialize_particles, motion_update, normalize_weights, get_mean_pose

sensor_data = load_sensor_data('../../data/sensor_data_clean.csv')
map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
## DON'T compute likelihood field here - do it in test function with correct sigma!

## CORRECTED STRATEGY - WIDER TOLERANCE FOR 20cm DRIFT
## Baseline (60b, n_min=1000, sigma=0.2) = 0.391m is BEST so far!
## Problem: sigma=0.1 was TOO TIGHT, killing good particles
## Solution: Test WIDER sigma for more tolerance
TEST_BEAMS = 90   # <<< CHANGE: 60, 90 (baseline winners)
TEST_SIGMA = 0.2   # <<< CHANGE: 0.25, 0.3, 0.35 (WIDER tolerance!)
TEST_N_MIN = 1000   # <<< CHANGE: 500, 750, 1000 (adaptive)

## BASELINE REFERENCES
BASELINE_NEW = 0.391  # 60 beams, n_min=1000, sigma=0.2 (NEW baseline winner!)
BEST_90B_ERROR = 0.415  # 90 beams, n_min=750
TIGHT_SIGMA_FAIL = 0.589  # 180b, sigma=0.1, n_min=1500 (WORSE - too strict!)
TARGET_ERROR = 0.15    # half the 20cm drift

print("="*60)
print("CORRECTED STRATEGY - WIDER SIGMA TOLERANCE")

UPDATE_SKIP = 10  # 5Hz updates

def test_combined_config(run_num, num_beams, sigma_hit, n_min):
    print(f"\n=== RUN {run_num}/3 ===")
    print(f"Computing likelihood field with sigma={sigma_hit}...")
    likelihood_field = compute_likelihood_field(map_info, sigma=sigma_hit, max_dist=2.0)

    ## override amcl params
    amcl.num_beams = num_beams
    amcl.sigma_hit = sigma_hit
    amcl.n_min = n_min

    ## pose tracking mode
    init_pose = tuple(sensor_data['ground_truth'][0])
    uncertainty = (0.3, 0.3, 0.5)
    particles = initialize_particles(500, map_info, init_pose, uncertainty)

    n_steps = len(sensor_data['timestamps'])
    timestamps = sensor_data['timestamps']
    results = []
    update_times = []

    print(f"Beams: {num_beams}, Sigma: {sigma_hit}, n_min: {n_min}")

    prev_t = 0

    for t in range(UPDATE_SKIP, n_steps, UPDATE_SKIP):
        start_time = time.time()

        ## USE ODOMETRY
        prev_odom = sensor_data['odometry'][prev_t]
        curr_odom = sensor_data['odometry'][t]

        motion_update(particles, prev_odom, curr_odom)

        ## sensor update
        lidar = sensor_data['lidar_scans'][t]
        amcl.sensor_update(particles, lidar, likelihood_field, map_info)
        normalize_weights(particles)

        ## KLD resample with current n_min
        particles = amcl.kld_resample(particles, map_info)

        est_x, est_y, est_theta = get_mean_pose(particles)
        gt = sensor_data['ground_truth'][t]

        err = np.sqrt((est_x - gt[0])**2 + (est_y - gt[1])**2)

        update_time = (time.time() - start_time) * 1000
        update_times.append(update_time)

        results.append({
            'timestamp': timestamps[t],
            'x_est': est_x,
            'y_est': est_y,
            'theta_est': est_theta,
            'x_gt': gt[0],
            'y_gt': gt[1],
            'theta_gt': gt[2],
            'error': err,
            'n_particles': len(particles),
            'update_time_ms': update_time
        })

        if len(results) % 25 == 0:
            print(f"t={timestamps[t]:.1f}s: err={err:.3f}m, n={len(particles)}, {update_time:.1f}ms")

        prev_t = t

    ## save detailed results
    with open(f'results/combined_b{num_beams}_s{sigma_hit}_n{n_min}_run{run_num}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'x_est', 'y_est', 'theta_est','x_gt', 'y_gt', 'theta_gt','error', 'n_particles', 'update_time_ms'])
        writer.writeheader()
        writer.writerows(results)

    ## analysis
    errors = [r['error'] for r in results]
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    std_err = np.std(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)

    ## convergence
    convergence_idx = 0
    window_size = 10
    for i in range(window_size, len(errors)):
        window_std = np.std(errors[i-window_size:i])
        if window_std < 0.1:
            convergence_idx = i
            break
    convergence_time = results[convergence_idx]['timestamp'] if convergence_idx > 0 else 0

    particles_count = [r['n_particles'] for r in results]
    mean_particles = np.mean(particles_count)
    std_particles = np.std(particles_count)

    mean_update_time = np.mean(update_times)
    max_update_time = np.max(update_times)

    print(f"Run {run_num} - Mean: {mean_err:.3f}m, Median: {median_err:.3f}m, Std: {std_err:.3f}m")
    print(f"           Particles: {mean_particles:.0f}±{std_particles:.0f}, Update: {mean_update_time:.1f}ms")

    return {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'min': min_err,
        'max': max_err,
        'particles_mean': mean_particles,
        'particles_std': std_particles,
        'convergence_time': convergence_time,
        'update_time_mean': mean_update_time,
        'update_time_max': max_update_time,
        'num_updates': len(results)
    }


## run test 3 times
all_runs = []
for run in range(1, 4):
    run_results = test_combined_config(run, TEST_BEAMS, TEST_SIGMA, TEST_N_MIN)
    all_runs.append(run_results)

## summary
print("\n" + "="*60)
print(f"COMBINED TEST RESULTS ({TEST_BEAMS}b, s={TEST_SIGMA}, n={TEST_N_MIN})")
print("="*60)
for i, r in enumerate(all_runs, 1):
    print(f"Run {i}: {r['mean']:.3f}m (median={r['median']:.3f}m, std={r['std']:.3f}m)")

## aggregate
avg_mean = np.mean([r['mean'] for r in all_runs])
avg_median = np.mean([r['median'] for r in all_runs])
avg_std = np.mean([r['std'] for r in all_runs])
consistency = np.std([r['mean'] for r in all_runs])

print(f"\nAverage across runs: {avg_mean:.3f}m")
print(f"Median across runs:  {avg_median:.3f}m")
print(f"Run consistency:     {consistency:.3f}m")
print(f"Avg convergence:     {np.mean([r['convergence_time'] for r in all_runs]):.1f}s")
print(f"Avg particles:       {np.mean([r['particles_mean'] for r in all_runs]):.0f}")
print(f"Avg update time:     {np.mean([r['update_time_mean'] for r in all_runs]):.1f}ms")

## improvement vs best configs
improvement_vs_baseline = ((BASELINE_NEW - avg_mean) / BASELINE_NEW) * 100
improvement_vs_90b = ((BEST_90B_ERROR - avg_mean) / BEST_90B_ERROR) * 100


## enhanced log
log_entry = {
    'test_name': f'combined_b{TEST_BEAMS}_s{TEST_SIGMA}_n{TEST_N_MIN}_NEW',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'dataset_info': {
        'trajectory_length': '5.79m',
        'duration': f"{sensor_data['timestamps'][-1]:.1f}s",
        'update_frequency': '5Hz'
    },
    'params': {
        'n_particles': 500,
        'alpha1': 0.005,
        'alpha2': 0.005,
        'alpha3': 0.02,
        'alpha4': 0.02,
        'z_hit': 0.95,
        'z_rand': 0.05,
        'sigma_hit': TEST_SIGMA,
        'num_beams': TEST_BEAMS,
        'n_min': TEST_N_MIN,
        'n_max': 5000,
        'epsilon': 0.05,
        'bin_size': 0.5
    },
    'results': {
        'run1_mean': round(all_runs[0]['mean'], 3),
        'run2_mean': round(all_runs[1]['mean'], 3),
        'run3_mean': round(all_runs[2]['mean'], 3),
        'average_mean': round(avg_mean, 3),
        'average_median': round(avg_median, 3),
        'average_std': round(avg_std, 3),
        'run_consistency': round(consistency, 3),
        'min_error': round(np.min([r['min'] for r in all_runs]), 3),
        'max_error': round(np.max([r['max'] for r in all_runs]), 3)
    },
    'performance': {
        'avg_particles': round(np.mean([r['particles_mean'] for r in all_runs]), 1),
        'particles_std': round(np.mean([r['particles_std'] for r in all_runs]), 1),
        'avg_convergence_time': round(np.mean([r['convergence_time'] for r in all_runs]), 2),
        'avg_update_time_ms': round(np.mean([r['update_time_mean'] for r in all_runs]), 2),
        'max_update_time_ms': round(np.max([r['update_time_max'] for r in all_runs]), 2)
    },
    'improvement_vs_baseline': round(improvement_vs_baseline, 1),
    'improvement_vs_90b_best': round(improvement_vs_90b, 1),
    'achieved_target': bool(avg_mean <= TARGET_ERROR),
    'notes': f'CORRECTED strategy: WIDER sigma tolerance for 20cm drift. Previous tight sigma (0.1) failed badly (0.589m). Testing wider sigma ({TEST_SIGMA}) with {TEST_BEAMS}b + n_min={TEST_N_MIN}. Baseline (60b, s=0.2, n=1000) = 0.391m is reference.'
}

## read existing log (keep all old entries!)
try:
    with open('calibration_log.json', 'r') as f:
        log = json.load(f)
except:
    log = []

log.append(log_entry)

with open('calibration_log.json', 'w') as f:
    json.dump(log, f, indent=2)
