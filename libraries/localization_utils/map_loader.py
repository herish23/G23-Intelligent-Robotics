# map loading from pgm/yaml files

import numpy as np
import yaml
from PIL import Image
from dataclasses import dataclass
from .geometry import world_to_grid


@dataclass
class MapInfo:
    # holds map data + metadata
    occupancy_grid: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    occupied_thresh: float
    free_thresh: float


def load_map(pgm_path, yaml_path):
    # loads occupancy grid from pgm image and yaml metadata
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    resolution = metadata['resolution']
    origin = metadata['origin']
    origin_x, origin_y = origin[0], origin[1]
    occupied_thresh = metadata.get('occupied_thresh', 0.65)
    free_thresh = metadata.get('free_thresh', 0.25)

    img = Image.open(pgm_path)
    img_array = np.array(img)

    # convert to occupancy: 0=free, 1=occupied, 0.5=unknown
    normalized = img_array / 255.0
    occupancy_grid = np.where(
        normalized > (1 - free_thresh),
        0.0,
        np.where(normalized < occupied_thresh, 1.0, 0.5)
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
    # check if position is in free space
    grid_x, grid_y = world_to_grid(x, y, map_info)

    if not (0 <= grid_x < map_info.width and 0 <= grid_y < map_info.height):
        return False

    if map_info.occupancy_grid[grid_y, grid_x] > 0.5:
        return False

    # check safety margin if needed
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
    # get map boundaries in world coords
    x_min = map_info.origin_x
    y_min = map_info.origin_y
    x_max = x_min + map_info.width * map_info.resolution
    y_max = y_min + map_info.height * map_info.resolution
    return x_min, x_max, y_min, y_max
