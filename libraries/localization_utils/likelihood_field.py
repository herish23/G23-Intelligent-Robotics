# liklihood field for sensor model
# used by markov and amcl for fast sensor updates

import numpy as np
from scipy.ndimage import distance_transform_edt
from .geometry import world_to_grid


def compute_likelihood_field(map_info, sigma=0.1, max_dist=2.0):
    # precompute likelihood around obstacles for quick lookups
    obstacles = (map_info.occupancy_grid > 0.5).astype(np.float32)
    dist_transform = distance_transform_edt(1 - obstacles)

    dist_meters = dist_transform * map_info.resolution
    max_dist_cells = max_dist / map_info.resolution
    sigma_cells = sigma / map_info.resolution

    # gaussian around obstacles
    likelihood = np.exp(-0.5 * (dist_transform / sigma_cells) ** 2)
    likelihood[dist_transform > max_dist_cells] = 0.01

    likelihood = likelihood / likelihood.max()
    return likelihood


def compute_beam_model_likelihood(scan_range, expected_range, sigma_hit=0.1,
                                    lambda_short=0.1, z_hit=0.7, z_short=0.1,
                                    z_max=0.1, z_rand=0.1, max_range=3.5):
    # beam model with hit/short/max/random mixture
    # more complex than liklihood field but maybe better

    if scan_range < max_range:
        p_hit = (1.0 / (sigma_hit * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * ((scan_range - expected_range) / sigma_hit) ** 2)
    else:
        p_hit = 0.0

    if 0 < scan_range < expected_range:
        p_short = lambda_short * np.exp(-lambda_short * scan_range)
    else:
        p_short = 0.0

    p_max = 1.0 if abs(scan_range - max_range) < 0.01 else 0.0
    p_rand = 1.0 / max_range if scan_range < max_range else 0.0

    return z_hit * p_hit + z_short * p_short + z_max * p_max + z_rand * p_rand


def ray_cast(x, y, theta, angle, map_info, max_range=3.5):
    # simple raycast to find expected range
    ray_angle = theta + angle
    resolution = map_info.resolution
    step_size = resolution / 2.0

    for dist in np.arange(0, max_range, step_size):
        px = x + dist * np.cos(ray_angle)
        py = y + dist * np.sin(ray_angle)
        gx, gy = world_to_grid(px, py, map_info)

        if not (0 <= gx < map_info.width and 0 <= gy < map_info.height):
            return max_range

        if map_info.occupancy_grid[gy, gx] > 0.5:
            return dist

    return max_range


def compute_scan_likelihood_field(scan_ranges, scan_angles, robot_x, robot_y,
                                    robot_theta, map_info, likelihood_field):
    # calc log-likelihood of scan using precomputed field (faster than raycasting)
    log_likelihood = 0.0
    num_valid = 0

    for r, a in zip(scan_ranges, scan_angles):
        if r < 0.12 or r > 3.5:
            continue

        global_angle = robot_theta + a
        px = robot_x + r * np.cos(global_angle)
        py = robot_y + r * np.sin(global_angle)

        gx, gy = world_to_grid(px, py, map_info)

        if 0 <= gx < map_info.width and 0 <= gy < map_info.height:
            p = max(likelihood_field[gy, gx], 1e-10)
            log_likelihood += np.log(p)
            num_valid += 1

    return log_likelihood / max(num_valid, 1)
