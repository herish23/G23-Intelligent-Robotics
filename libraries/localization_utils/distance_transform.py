# distance tranform utils for EKF measurement model

import numpy as np
from scipy.ndimage import distance_transform_edt
from .geometry import world_to_grid


def compute_distance_transform(map_info):
    # euclidean dist transform - returns distance in grid cells
    free_space = (map_info.occupancy_grid < 0.5).astype(np.float32)
    dist_transform = distance_transform_edt(free_space)
    return dist_transform


def compute_gradient_map(dist_transform):
    # gradient for jacobian computation
    grad_y, grad_x = np.gradient(dist_transform)
    return grad_x, grad_y


def chamfer_distance(scan_points, map_info, dist_transform):
    # chamfer dist - sum of distances from scan points to obstacles
    total_dist = 0.0
    num_valid = 0

    for point in scan_points:
        x, y = point
        gx, gy = world_to_grid(x, y, map_info)

        if 0 <= gx < map_info.width and 0 <= gy < map_info.height:
            dist_cells = dist_transform[gy, gx]
            dist_meters = dist_cells * map_info.resolution
            total_dist += dist_meters
            num_valid += 1

    return total_dist / max(num_valid, 1)


def bilinear_interpolate(array, x, y):
    # bilinear interp for non-integer positions
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    height, width = array.shape
    # clip to bounds
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
