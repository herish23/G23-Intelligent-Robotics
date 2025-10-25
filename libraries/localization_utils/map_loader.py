"""
Map loading utilities for occupancy grid maps
"""

import numpy as np
import yaml
from PIL import Image
from dataclasses import dataclass


@dataclass
class MapInfo:
    """Container for map data and metadata"""
    occupancy_grid: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    occupied_thresh: float
    free_thresh: float


def load_map(pgm_path, yaml_path):
    """Load occupancy grid map from PGM and YAML files"""
    # Load metadata
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    resolution = metadata['resolution']
    origin = metadata['origin']
    origin_x, origin_y = origin[0], origin[1]
    occupied_thresh = metadata.get('occupied_thresh', 0.65)
    free_thresh = metadata.get('free_thresh', 0.25)
    negate = metadata.get('negate', 0)

    # Load image
    img = Image.open(pgm_path)
    img_array = np.array(img)

    # Convert to occupancy probabilities
    # PGM: 255=white=free, 0=black=occupied
    normalized = img_array / 255.0
    occupancy_grid = np.where(
        normalized > (1 - free_thresh),
        0.0,  # Free
        np.where(normalized < occupied_thresh, 1.0, 0.5)  # Occupied or Unknown
    )

    height, width = occupancy_grid.shape

    return MapInfo(
        occupancy_grid=occupancy_grid,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh
    )


def is_valid_position(x, y, map_info, safety_margin=0.0):
    """Check if world position is in free space"""
    from .geometry import world_to_grid

    grid_x, grid_y = world_to_grid(x, y, map_info)

    # Check bounds
    if not (0 <= grid_x < map_info.width and 0 <= grid_y < map_info.height):
        return False

    # Check if occupied
    if map_info.occupancy_grid[grid_y, grid_x] > 0.5:
        return False

    # Optional safety margin check
    if safety_margin > 0:
        margin_cells = int(safety_margin / map_info.resolution)
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                if 0 <= gx < map_info.width and 0 <= gy < map_info.height:
                    if map_info.occupancy_grid[gy, gx] > 0.5:
                        return False

    return True


def get_map_bounds(map_info):
    """Get world coordinates of map boundaries"""
    x_min = map_info.origin_x
    y_min = map_info.origin_y
    x_max = x_min + map_info.width * map_info.resolution
    y_max = y_min + map_info.height * map_info.resolution
    return x_min, x_max, y_min, y_max
