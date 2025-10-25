"""
Distance transform for EKF-OGM measurement model
Based on Dantanarayana et al. (2015)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_distance_transform(map_info):
    """
    Compute distance from each cell to nearest obstacle
    Returns distance in grid cells (multiply by resolution for meters)
    """
    free_space = (map_info.occupancy_grid < 0.5).astype(np.float32)
    dist_transform = distance_transform_edt(free_space)
    return dist_transform


def compute_gradient_map(dist_transform):
    """Compute gradient of distance transform for Jacobians"""
    grad_y, grad_x = np.gradient(dist_transform)
    return grad_x, grad_y


def chamfer_distance(scan_points, map_info, dist_transform):
    """
    Compute Chamfer distance for LiDAR scan
    Sum of distances from each point to nearest obstacle
    """
    from .geometry import world_to_grid

    total_dist = 0.0
    num_valid = 0

    for point in scan_points:
        x, y = point[0], point[1]
        gx, gy = world_to_grid(x, y, map_info)

        if 0 <= gx < map_info.width and 0 <= gy < map_info.height:
            dist_cells = dist_transform[gy, gx]
            dist_meters = dist_cells * map_info.resolution
            total_dist += dist_meters
            num_valid += 1

    return total_dist / max(num_valid, 1)


def bilinear_interpolate(array, x, y):
    """Interpolate array value at non-integer position"""
    x0, x1 = int(np.floor(x)), int(np.floor(x)) + 1
    y0, y1 = int(np.floor(y)), int(np.floor(y)) + 1

    height, width = array.shape
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    wx, wy = x - x0, y - y0

    value = (
        array[y0, x0] * (1 - wx) * (1 - wy) +
        array[y0, x1] * wx * (1 - wy) +
        array[y1, x0] * (1 - wx) * wy +
        array[y1, x1] * wx * wy
    )
    return value
