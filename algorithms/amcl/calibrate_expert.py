import sys
import numpy as np
import csv
import json
import time
from datetime import datetime

sys.path.append("../../libraries")
from localization_utils import load_sensor_data, load_map, compute_likelihood_field

## import from amcl but override params
import amcl
from amcl import initialize_particles, normalize_weights, get_mean_pose

sensor_data = load_sensor_data('../../data/sensor_data_clean.csv')
map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')


print("="*60)

## EXPERT-RECOMMENDED PARAMS
EXPERT_BEAMS = 270        # 75% of 360 beams (every ~1.3 degrees)
EXPERT_SIGMA = 0.25       # Slightly softer than 0.2
EXPERT_MAX_RANGE = 1.8    # Matches your actual sensor data
EXPERT_Z_HIT = 0.85       # Less overconfident
EXPERT_Z_RAND = 0.15      # More robust to outliers

## KLD params - slightly tighter
EXPERT_N_MIN = 1000
EXPERT_N_MAX = 5000
EXPERT_EPSILON = 0.04     # Tighter than 0.05
EXPERT_BIN_SIZE = 0.6     # Coarser than 0.5

## Update rate - expert suggests 10Hz if CPU allows
EXPERT_UPDATE_SKIP = 5    # 10Hz instead of 5Hz

## Motion model - keep default for now
EXPERT_ALPHA1 = 0.005
EXPERT_ALPHA2 = 0.005
EXPERT_ALPHA3 = 0.02
EXPERT_ALPHA4 = 0.02

## Baseline reference
BASELINE_BEST = 0.384  # from amcl.py

## Compute likelihood field with expert sigma
likelihood_field = compute_likelihood_field(map_info, sigma=EXPERT_SIGMA, max_dist=2.0)

## N_eff calculation (expert recommendation)
def compute_neff(particles):
    """Effective sample size - measures particle degeneracy"""
    weights = np.array([p.weight for p in particles])
    return 1.0 / np.sum(weights**2) if np.sum(weights**2) > 0 else len(particles)

## Custom sensor update with expert params
def expert_sensor_update(particles, lidar_ranges, likelihood_field, map_info):
    n_total = len(lidar_ranges)
    angle_inc = 2 * np.pi / n_total

    # Sample EXPERT_BEAMS uniformly
    beam_step = max(1, n_total // EXPERT_BEAMS)

    eps = 1e-12
    for p in particles:
        log_w = 0.0
        n_beams = 0

        for i in range(0, n_total, beam_step):
            z = lidar_ranges[i]

            # Skip invalid or max-range readings
            if not np.isfinite(z) or z >= EXPERT_MAX_RANGE:
                continue

            angle = p.theta + (i * angle_inc - np.pi)  # assuming angle_min = -pi
            hit_x = p.x + z * np.cos(angle)
            hit_y = p.y + z * np.sin(angle)

            mx = int((hit_x - map_info.origin_x) / map_info.resolution)
            my = int((hit_y - map_info.origin_y) / map_info.resolution)

            if 0 <= mx < map_info.width and 0 <= my < map_info.height:
                prob_hit = likelihood_field[my, mx]
                prob_z = EXPERT_Z_HIT * prob_hit + EXPERT_Z_RAND / EXPERT_MAX_RANGE
                log_w += np.log(max(prob_z, eps))
            else:
                log_w += np.log(eps)

            n_beams += 1

        # Average log weight
        if n_beams > 0:
            p.weight = np.exp(log_w / n_beams)
        else:
            p.weight = eps

## Expert KLD with tuned params
def expert_kld_resample(particles, map_info, n_min, n_max, epsilon, bin_size):
    n = len(particles)

    # Normalize weights
    w_sum = sum(p.weight for p in particles)
    if w_sum == 0:
        for p in particles:
            p.weight = 1.0 / n
        return particles

    for p in particles:
        p.weight /= w_sum

    # Build cumulative weights
    cum_w = []
    c = 0.0
    for p in particles:
        c += p.weight
        cum_w.append(c)

    new_p = []
    bins = {}
    k = 0
    M_x = 0

    while True:
        if k > 1:
            z_quantile = 2.58  # 99% confidence
            thresh = (k - 1) / (2 * epsilon) * (1 - 2/(9*(k-1)) + np.sqrt(2/(9*(k-1))) * z_quantile)**3
            if M_x >= thresh and M_x >= n_min:
                break

        if M_x >= n_max:
            break

        # Sample particle
        r = np.random.uniform(0, 1.0)
        idx = 0
        for i in range(n):
            if cum_w[i] >= r:
                idx = i
                break

        p_new = amcl.Particle(particles[idx].x, particles[idx].y, particles[idx].theta, 0.0)
        new_p.append(p_new)
        M_x += 1

        # Bin with expert bin_size
        bx = int(np.floor((p_new.x - map_info.origin_x) / bin_size))
        by = int(np.floor((p_new.y - map_info.origin_y) / bin_size))
        theta_wrap = p_new.theta + np.pi
        btheta = int(np.floor(theta_wrap / (np.pi/4))) % 8
        key = (bx, by, btheta)

        if key not in bins:
            k += 1
            bins[key] = True

    # Uniform weights
    for p in new_p:
        p.weight = 1.0 / len(new_p)

    return new_p

## Test function
def test_expert_config(run_num):
    ## Override amcl params
    amcl.num_beams = EXPERT_BEAMS
    amcl.sigma_hit = EXPERT_SIGMA
    amcl.z_hit = EXPERT_Z_HIT
    amcl.z_rand = EXPERT_Z_RAND
    amcl.alpha1 = EXPERT_ALPHA1
    amcl.alpha2 = EXPERT_ALPHA2
    amcl.alpha3 = EXPERT_ALPHA3
    amcl.alpha4 = EXPERT_ALPHA4

    ## Pose tracking mode
    init_pose = tuple(sensor_data['ground_truth'][0])
    uncertainty = (0.3, 0.3, 0.5)
    particles = initialize_particles(500, map_info, init_pose, uncertainty)

    n_steps = len(sensor_data['timestamps'])
    timestamps = sensor_data['timestamps']
    results = []
    update_times = []

    print(f"\n=== RUN {run_num}/3 ===")
    prev_t = 0
    neff_resample_count = 0

    for t in range(EXPERT_UPDATE_SKIP, n_steps, EXPERT_UPDATE_SKIP):
        start_time = time.time()

        ## Motion update with odometry
        prev_odom = sensor_data['odometry'][prev_t]
        curr_odom = sensor_data['odometry'][t]
        amcl.motion_update(particles, prev_odom, curr_odom)

        ## Expert sensor update
        lidar = sensor_data['lidar_scans'][t]
        expert_sensor_update(particles, lidar, likelihood_field, map_info)
        normalize_weights(particles)

        ## N_eff adaptive resampling (EXPERT RECOMMENDATION)
        neff = compute_neff(particles)
        neff_threshold = 0.6 * len(particles)

        if neff < neff_threshold:
            particles = expert_kld_resample(particles, map_info, EXPERT_N_MIN, EXPERT_N_MAX,
                                           EXPERT_EPSILON, EXPERT_BIN_SIZE)
            neff_resample_count += 1

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
            'neff': neff,
            'update_time_ms': update_time
        })

        if len(results) % 50 == 0:
            print(f"t={timestamps[t]:.1f}s: err={err:.3f}m, n={len(particles)}, N_eff={neff:.0f}, {update_time:.1f}ms")

        prev_t = t

    ## Save results
    with open(f'results/expert_run{run_num}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'x_est', 'y_est', 'theta_est','x_gt', 'y_gt', 'theta_gt', 'error','n_particles', 'neff', 'update_time_ms'])
        writer.writeheader()
        writer.writerows(results)

    ## Analysis
    errors = [r['error'] for r in results]
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    std_err = np.std(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)

    mean_particles = np.mean([r['n_particles'] for r in results])
    std_particles = np.std([r['n_particles'] for r in results])
    mean_neff = np.mean([r['neff'] for r in results])
    mean_update_time = np.mean(update_times)

    print(f"Run {run_num} - Mean: {mean_err:.3f}m, Median: {median_err:.3f}m, Std: {std_err:.3f}m")
    print(f"           Particles: {mean_particles:.0f}±{std_particles:.0f}, N_eff: {mean_neff:.0f}")
    print(f"           Resampled: {neff_resample_count}/{len(results)} times ({100*neff_resample_count/len(results):.1f}%)")

    return {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'min': min_err,
        'max': max_err,
        'particles_mean': mean_particles,
        'particles_std': std_particles,
        'neff_mean': mean_neff,
        'resample_count': neff_resample_count,
        'update_time_mean': mean_update_time
    }

## Run 3 tests
all_runs = []
for run in range(1, 4):
    run_results = test_expert_config(run)
    all_runs.append(run_results)

## Summary
for i, r in enumerate(all_runs, 1):
    print(f"Run {i}: {r['mean']:.3f}m (median={r['median']:.3f}m, std={r['std']:.3f}m)")

avg_mean = np.mean([r['mean'] for r in all_runs])
avg_median = np.mean([r['median'] for r in all_runs])
consistency = np.std([r['mean'] for r in all_runs])

print(f"\nAverage across runs: {avg_mean:.3f}m")
print(f"Median across runs:  {avg_median:.3f}m")
print(f"Run consistency:     {consistency:.3f}m (lower is better)")
print(f"Avg N_eff:           {np.mean([r['neff_mean'] for r in all_runs]):.0f}")
print(f"Avg resample rate:   {np.mean([r['resample_count'] for r in all_runs]):.0f} times")

improvement = ((BASELINE_BEST - avg_mean) / BASELINE_BEST) * 100
print(f"\nBaseline (amcl.py default): {BASELINE_BEST:.3f}m")
print(f"Expert config:              {avg_mean:.3f}m")
print(f"Improvement: {improvement:+.1f}%")
print("="*60)

## Log
log_entry = {
    'test_name': 'expert_recommended',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'params': {
        'num_beams': EXPERT_BEAMS,
        'sigma_hit': EXPERT_SIGMA,
        'max_range': EXPERT_MAX_RANGE,
        'z_hit': EXPERT_Z_HIT,
        'z_rand': EXPERT_Z_RAND,
        'n_min': EXPERT_N_MIN,
        'n_max': EXPERT_N_MAX,
        'epsilon': EXPERT_EPSILON,
        'bin_size': EXPERT_BIN_SIZE,
        'update_skip': EXPERT_UPDATE_SKIP,
        'neff_threshold': 0.6,
        'alpha1': EXPERT_ALPHA1,
        'alpha2': EXPERT_ALPHA2,
        'alpha3': EXPERT_ALPHA3,
        'alpha4': EXPERT_ALPHA4
    },
    'results': {
        'run1_mean': round(all_runs[0]['mean'], 3),
        'run2_mean': round(all_runs[1]['mean'], 3),
        'run3_mean': round(all_runs[2]['mean'], 3),
        'average_mean': round(avg_mean, 3),
        'average_median': round(avg_median, 3),
        'run_consistency': round(consistency, 3),
        'min_error': round(np.min([r['min'] for r in all_runs]), 3),
        'max_error': round(np.max([r['max'] for r in all_runs]), 3)
    },
    'improvement_vs_baseline': round(improvement, 1),
    'notes': 'Expert-recommended config: 270 beams, max_range=1.8m, z_hit=0.85/0.15, sigma=0.25m, 10Hz updates, N_eff adaptive resampling with 0.6 threshold'
}

try:
    with open('calibration_log.json', 'r') as f:
        log = json.load(f)
except:
    log = []

log.append(log_entry)

with open('calibration_log.json', 'w') as f:
    json.dump(log, f, indent=2)

