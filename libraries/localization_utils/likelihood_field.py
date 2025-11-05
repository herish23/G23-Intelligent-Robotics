import numpy as np
from scipy.ndimage import distance_transform_edt


## precompute likelihood field for sensor model
def compute_likelihood_field(map_info, sigma=0.1, max_dist=2.0):
    # binary obstacle map
    obs = (map_info.occupancy_grid > 0.5).astype(np.float32)

    # distance transform - euclidean dist to nearest obstacle
    dist = distance_transform_edt(1 - obs)

    # convert to meters
    dist_m = dist * map_info.resolution

    # gaussian around obstacles
    sig_cells = sigma / map_info.resolution
    field = np.exp(-0.5 * (dist / sig_cells) ** 2)

    # clamp far ditsances
    max_cells = max_dist / map_info.resolution
    field[dist > max_cells] = 0.01

    # normalize
    field = field / field.max()

    return field
