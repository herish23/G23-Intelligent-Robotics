"""
Configuration loading from YAML files
"""

import yaml
import os


def load_config(config_path):
    """Load experiment configuration from YAML"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config):
    """Check required fields exist"""
    required = ['experiment', 'data', 'robot', 'sensors']
    for field in required:
        if field not in config:
            raise ValueError(f"Config missing '{field}' section")

    if 'lidar' not in config['sensors'] or 'odometry' not in config['sensors']:
        raise ValueError("Config missing sensor specifications")

    if len(config['robot']['initial_pose']) != 3:
        raise ValueError("initial_pose must be [x, y, theta]")

    if len(config['sensors']['odometry']['alphas']) != 4:
        raise ValueError("odometry alphas must have 4 values")


def get_output_paths(config, algorithm_name):
    """Generate output file paths for algorithm results"""
    exp_name = config['experiment']['name']
    output_dir = config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    return {
        'estimate_file': os.path.join(output_dir, f"{algorithm_name}_estimate_{exp_name}.csv"),
        'timing_file': os.path.join(output_dir, f"{algorithm_name}_timing_{exp_name}.csv")
    }


def print_config_summary(config):
    """Print configuration summary"""
    print(f"Experiment: {config['experiment']['name']}")
    if 'description' in config['experiment']:
        print(f"  {config['experiment']['description']}")
    print(f"\nSensor Configuration:")
    print(f"  LiDAR noise: {config['sensors']['lidar']['noise_std']} m")
    print(f"  LiDAR sparsity: 1/{config['sensors']['lidar']['sparsity']}")
    print(f"  Odometry alphas: {config['sensors']['odometry']['alphas']}")
    print(f"\nInitial pose offset: {config['robot']['initial_pose']}")
    print(f"Random seed: {config.get('random_seed', 42)}")
