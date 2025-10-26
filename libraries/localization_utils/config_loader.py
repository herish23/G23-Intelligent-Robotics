# load experiment config from yaml

import yaml
import os


def load_config(config_path):
    # load and validate config file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    return config


def _validate_config(config):
    # check required fields are present
    required = ['experiment', 'data', 'robot', 'sensors']
    for field in required:
        if field not in config:
            raise ValueError(f"missing {field} in config")

    if 'lidar' not in config['sensors'] or 'odometry' not in config['sensors']:
        raise ValueError("sensor config incomplete")

    if len(config['robot']['initial_pose']) != 3:
        raise ValueError("initial_pose needs [x, y, theta]")

    if len(config['sensors']['odometry']['alphas']) != 4:
        raise ValueError("alphas need 4 values")


def get_output_paths(config, algorithm_name):
    # generate output file paths
    exp_name = config['experiment']['name']
    output_dir = config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    return {
        'estimate_file': os.path.join(output_dir, f"{algorithm_name}_estimate_{exp_name}.csv"),
        'timing_file': os.path.join(output_dir, f"{algorithm_name}_timing_{exp_name}.csv")
    }


def print_config_summary(config):
    # print experiment setup
    print(f"Experiment: {config['experiment']['name']}")
    if 'description' in config['experiment']:
        print(f"  {config['experiment']['description']}")
    print(f"\nSensor Configuration:")
    print(f"  LiDAR noise: {config['sensors']['lidar']['noise_std']} m")
    print(f"  LiDAR sparsity: 1/{config['sensors']['lidar']['sparsity']}")
    print(f"  Odometry alphas: {config['sensors']['odometry']['alphas']}")
    print(f"\nInitial pose offset: {config['robot']['initial_pose']}")
    print(f"Random seed: {config.get('random_seed', 42)}")
