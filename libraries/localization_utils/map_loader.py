"""
Map Loading Utilities

Loads occupancy grid maps from PGM and YAML files.
"""

import numpy as np
import yaml
from PIL import Image
from dataclasses import dataclass


@dataclass
class MapInfo:
    """
    Container for map metadata

    Attributes:
        occupancy_grid (np.ndarray): 2D array where 0=free, 1=occupied
        resolution (float): Meters per pixel
        origin_x (float): X coordinate of map origin (bottom-left) in meters
        origin_y (float): Y coordinate of map origin (bottom-left) in meters
        width (int): Map width in pixels
        height (int): Map height in pixels
        occupied_thresh (float): Threshold for occupied cells
        free_thresh (float): Threshold for free cells
    """
    occupancy_grid: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    occupied_thresh: float
    free_thresh: float


def load_map(pgm_path, yaml_path):
    """
    Load occupancy grid map from PGM and YAML files

    Args:
        pgm_path (str): Path to .pgm image file
        yaml_path (str): Path to .yaml metadata file

    Returns:
        MapInfo: Map data and metadata

    Example:
        >>> map_info = load_map("maps/epuck_world_map.pgm", "maps/epuck_world_map.yaml")
        >>> print(f"Map size: {map_info.width}x{map_info.height}")
        Map size: 300x300
        >>> print(f"Resolution: {map_info.resolution} m/pixel")
        Resolution: 0.01 m/pixel
    """
    # Load YAML metadata
    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    resolution = metadata['resolution']
    origin = metadata['origin']
    origin_x, origin_y = origin[0], origin[1]
    occupied_thresh = metadata.get('occupied_thresh', 0.65)
    free_thresh = metadata.get('free_thresh', 0.25)
    negate = metadata.get('negate', 0)

    # Load PGM image
    img = Image.open(pgm_path)
    img_array = np.array(img)

    # Convert to occupancy grid (0=free, 1=occupied)
    # PGM: 255=white=free, 0=black=occupied
    if negate:
        # If negate=1, invert: 0=free, 255=occupied
        occupancy_grid = (img_array < 128).astype(np.float32)
    else:
        # If negate=0 (default): 255=free, 0=occupied
        occupancy_grid = (img_array < 128).astype(np.float32)

    # Handle trinary mode (free, occupied, unknown)
    # Convert to probability: 0.0=free, 1.0=occupied, 0.5=unknown
    normalized = img_array / 255.0
    occupancy_grid = np.where(
        normalized > (1 - free_thresh),
        0.0,  # Free space
        np.where(
            normalized < occupied_thresh,
            1.0,  # Occupied
            0.5   # Unknown
        )
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
    """
    Check if a world position is valid (free space)

    Args:
        x (float): X coordinate in world frame (meters)
        y (float): Y coordinate in world frame (meters)
        map_info (MapInfo): Map metadata
        safety_margin (float): Extra clearance around obstacles (meters)

    Returns:
        bool: True if position is in free space

    Example:
        >>> is_valid_position(0.0, 0.0, map_info)
        True  # Center of arena is free
        >>> is_valid_position(1.5, 1.5, map_info)
        False  # Near wall, occupied
    """
    from .geometry import world_to_grid

    grid_x, grid_y = world_to_grid(x, y, map_info)

    # Check bounds
    if not (0 <= grid_x < map_info.width and 0 <= grid_y < map_info.height):
        return False

    # Check if occupied
    if map_info.occupancy_grid[grid_y, grid_x] > 0.5:
        return False

    # Check safety margin if specified
    if safety_margin > 0:
        margin_cells = int(safety_margin / map_info.resolution)
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                gx = grid_x + dx
                gy = grid_y + dy
                if 0 <= gx < map_info.width and 0 <= gy < map_info.height:
                    if map_info.occupancy_grid[gy, gx] > 0.5:
                        return False

    return True


def get_map_bounds(map_info):
    """
    Get world coordinates of map boundaries

    Args:
        map_info (MapInfo): Map metadata

    Returns:
        tuple: (x_min, x_max, y_min, y_max) in meters
    """
    x_min = map_info.origin_x
    y_min = map_info.origin_y
    x_max = x_min + map_info.width * map_info.resolution
    y_max = y_min + map_info.height * map_info.resolution
    return x_min, x_max, y_min, y_max
