"""
Geometry utilities for coordinate transformations and angle operations
"""

import numpy as np


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def world_to_grid(x, y, map_info):
    """Convert world coordinates (meters) to grid coordinates (pixels)"""
    grid_x = int((x - map_info.origin_x) / map_info.resolution)
    grid_y = int((y - map_info.origin_y) / map_info.resolution)
    return grid_x, grid_y


def grid_to_world(grid_x, grid_y, map_info):
    """Convert grid coordinates (pixels) to world coordinates (meters)"""
    x = grid_x * map_info.resolution + map_info.origin_x
    y = grid_y * map_info.resolution + map_info.origin_y
    return x, y


def rotation_matrix(theta):
    """2D rotation matrix"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s,  c]])


def transform_scan_to_world(ranges, angles, robot_x, robot_y, robot_theta, max_range=3.5):
    """Transform LiDAR scan from robot frame to world frame"""
    # Filter valid ranges
    valid = (ranges > 0.12) & (ranges < max_range)
    valid_ranges = ranges[valid]
    valid_angles = angles[valid]

    # Convert to Cartesian in robot frame
    x_robot = valid_ranges * np.cos(valid_angles)
    y_robot = valid_ranges * np.sin(valid_angles)

    # Rotate and translate to world frame
    R = rotation_matrix(robot_theta)
    points = R @ np.vstack([x_robot, y_robot])
    points[0, :] += robot_x
    points[1, :] += robot_y

    return points.T


def compute_distance(x1, y1, x2, y2):
    """Euclidean distance between two points"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def angle_difference(angle1, angle2):
    """Smallest difference between two angles"""
    return wrap_angle(angle1 - angle2)
