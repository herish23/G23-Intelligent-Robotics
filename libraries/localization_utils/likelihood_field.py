"""
Likelihood Field Utilities
Used by Markov Localization and AMCL for efficient sensor updates.

"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def compute_likelihood_field(map_info, sigma=0.1, max_dist=2.0):
    """
    Compute likelihood field for range sensor model

    The likelihood field represents the probability of a range measurement
    hitting an obstacle at each map location. Higher values near obstacles.

    Args:
        map_info (MapInfo): Map data from map_loader
        sigma (float): Standard deviation of Gaussian around obstacles (meters)
        max_dist (float): Maximum distance to consider (meters)

    Returns:
        np.ndarray: Likelihood field (same shape as occupancy_grid)
                    Values in range [0, 1]

    Example:
        >>> map_info = load_map("maps/epuck_world_map.pgm", "maps/epuck_world_map.yaml")
        >>> likelihood = compute_likelihood_field(map_info, sigma=0.1)
        >>> # High likelihood near obstacles, low in open space
        >>> print(f"Likelihood at center: {likelihood[150, 150]:.3f}")
    """
    # Create binary map: 1 where occupied, 0 where free
    obstacles = (map_info.occupancy_grid > 0.5).astype(np.float32)

    # Compute distance transform (distance to nearest obstacle)
    dist_transform = distance_transform_edt(1 - obstacles)

    # Convert distance from cells to meters
    dist_meters = dist_transform * map_info.resolution

    # Convert max_dist to cells
    max_dist_cells = max_dist / map_info.resolution

    # Compute Gaussian likelihood around obstacles
    # p(z | x, m) ∝ exp(-dist² / (2σ²))
    sigma_cells = sigma / map_info.resolution
    likelihood = np.exp(-0.5 * (dist_transform / sigma_cells) ** 2)

    # Clamp far distances to small probability
    likelihood[dist_transform > max_dist_cells] = 0.01

    # Normalize to [0, 1]
    likelihood = likelihood / likelihood.max()

    return likelihood


def compute_beam_model_likelihood(scan_range, expected_range, sigma_hit=0.1, lambda_short=0.1,
                                    z_hit=0.7, z_short=0.1, z_max=0.1, z_rand=0.1, max_range=3.5):
    """
    Compute likelihood using beam sensor model

    More sophisticated than likelihood field - models different failure modes:
    - Hits: Gaussian around expected range
    - Short: Exponential (unexpected obstacles)
    - Max: Uniform (max range failures)
    - Random: Uniform noise

    Args:
        scan_range (float): Measured range (meters)
        expected_range (float): Expected range from ray-tracing (meters)
        sigma_hit (float): Std dev for hit distribution
        lambda_short (float): Parameter for short distribution
        z_hit, z_short, z_max, z_rand (float): Mixture weights (must sum to 1.0)
        max_range (float): Maximum sensor range

    Returns:
        float: Likelihood p(z | expected_range)

    Example:
        >>> # Measured 1.5m, expected 1.5m → high likelihood
        >>> p1 = compute_beam_model_likelihood(1.5, 1.5)
        >>> print(f"Likelihood (match): {p1:.3f}")
        >>> # Measured 1.5m, expected 2.0m → lower likelihood
        >>> p2 = compute_beam_model_likelihood(1.5, 2.0)
        >>> print(f"Likelihood (mismatch): {p2:.3f}")
    """
    # Component 1: Hit (Gaussian around expected)
    if scan_range < max_range:
        p_hit = (1.0 / (sigma_hit * np.sqrt(2 * np.pi))) * \
                np.exp(-0.5 * ((scan_range - expected_range) / sigma_hit) ** 2)
    else:
        p_hit = 0.0

    # Component 2: Short (exponential decay for early hits)
    if 0 < scan_range < expected_range:
        p_short = lambda_short * np.exp(-lambda_short * scan_range)
    else:
        p_short = 0.0

    # Component 3: Max range failures
    if abs(scan_range - max_range) < 0.01:
        p_max = 1.0
    else:
        p_max = 0.0

    # Component 4: Random measurements
    p_rand = 1.0 / max_range if scan_range < max_range else 0.0

    # Mixture model
    p_total = z_hit * p_hit + z_short * p_short + z_max * p_max + z_rand * p_rand

    return p_total


def ray_cast(x, y, theta, angle, map_info, max_range=3.5):
    """
    Cast a ray in the map to find expected range

    Simple ray-tracing for computing expected LiDAR measurements.

    Args:
        x, y (float): Robot position in world frame
        theta (float): Robot orientation (radians)
        angle (float): Ray angle relative to robot (radians)
        map_info (MapInfo): Map metadata
        max_range (float): Maximum ray length

    Returns:
        float: Expected range to nearest obstacle (meters)

    Example:
        >>> # Ray at 0° from robot at origin
        >>> expected = ray_cast(0.0, 0.0, 0.0, 0.0, map_info)
        >>> print(f"Expected range: {expected:.2f} m")
    """
    from .geometry import world_to_grid

    # Global ray angle
    ray_angle = theta + angle

    # Step along ray
    resolution = map_info.resolution
    step_size = resolution / 2.0  # Half resolution for accuracy

    for dist in np.arange(0, max_range, step_size):
        # Current point along ray
        px = x + dist * np.cos(ray_angle)
        py = y + dist * np.sin(ray_angle)

        # Convert to grid
        gx, gy = world_to_grid(px, py, map_info)

        # Check bounds
        if not (0 <= gx < map_info.width and 0 <= gy < map_info.height):
            return max_range

        # Check if hit obstacle
        if map_info.occupancy_grid[gy, gx] > 0.5:
            return dist

    return max_range


def compute_scan_likelihood_field(scan_ranges, scan_angles, robot_x, robot_y, robot_theta,
                                    map_info, likelihood_field):
    """
    Compute total likelihood of a scan using pre-computed likelihood field

    Faster than ray-tracing - just lookup pre-computed field.

    Args:
        scan_ranges (np.ndarray): Range measurements (meters)
        scan_angles (np.ndarray): Corresponding angles (radians, robot frame)
        robot_x, robot_y, robot_theta (float): Robot pose
        map_info (MapInfo): Map metadata
        likelihood_field (np.ndarray): Pre-computed likelihood field

    Returns:
        float: Log-likelihood of entire scan

    Example:
        >>> likelihood_field = compute_likelihood_field(map_info)
        >>> log_p = compute_scan_likelihood_field(ranges, angles, x, y, theta,
        ...                                        map_info, likelihood_field)
        >>> print(f"Scan log-likelihood: {log_p:.2f}")
    """
    from .geometry import world_to_grid

    log_likelihood = 0.0
    num_valid = 0

    for i, (r, a) in enumerate(zip(scan_ranges, scan_angles)):
        # Skip invalid ranges
        if r < 0.12 or r > 3.5:
            continue

        # Endpoint of ray in world frame
        global_angle = robot_theta + a
        px = robot_x + r * np.cos(global_angle)
        py = robot_y + r * np.sin(global_angle)

        # Convert to grid
        gx, gy = world_to_grid(px, py, map_info)

        # Check bounds
        if 0 <= gx < map_info.width and 0 <= gy < map_info.height:
            # Lookup pre-computed likelihood
            p = likelihood_field[gy, gx]
            # Avoid log(0)
            p = max(p, 1e-10)
            log_likelihood += np.log(p)
            num_valid += 1

    # Normalize by number of valid beams
    if num_valid > 0:
        log_likelihood /= num_valid

    return log_likelihood
