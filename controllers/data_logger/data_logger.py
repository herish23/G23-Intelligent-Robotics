"""
Data Logger Controller
Collects clean sensor data (LiDAR + odometry + ground truth) for offline processing
"""

from controller import Supervisor
import csv
import math
import os

# Configuration
ROBOT_DEF = os.environ.get("ROBOT_DEF", "TB3")
ROBOT_NAME = os.environ.get("ROBOT_NAME", "TurtleBot3Burger")
TRAJ_PATH = os.environ.get("TRAJ", "../../maps/traj/traj_eval_complex.csv")
OUT_DIR = os.environ.get("OUT_DIR", "../../data")
OUT_FILE = os.environ.get("OUT_FILE", "sensor_data_clean.csv")
DT_LOG = float(os.environ.get("DT_LOG", "0.02"))  # 20ms

# Robot parameters
WHEEL_BASE = 0.160
WHEEL_RADIUS = 0.033


def yaw_from_axis_angle(ax, ay, az, ang):
    """Convert axis-angle to yaw"""
    s = math.sin(ang / 2.0)
    c = math.cos(ang / 2.0)
    qx, qy, qz, qw = ax * s, ay * s, az * s, c
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def load_trajectory(path):
    """Load trajectory from CSV"""
    with open(path) as f:
        reader = csv.DictReader(f)
        traj = [(float(r['timestamp']),
                 float(r['target_linear_velocity']),
                 float(r['target_angular_velocity'])) for r in reader]
    traj.sort(key=lambda x: x[0])
    return traj


def find_robot(sup):
    """Find robot node in scene"""
    node = sup.getFromDef(ROBOT_DEF)
    if node:
        return node

    # Fallback: search by name
    root = sup.getRoot()
    queue = [root]
    while queue:
        n = queue.pop(0)
        try:
            name_field = n.getField("name")
            if name_field and name_field.getSFString() == ROBOT_NAME:
                return n
        except:
            pass
        try:
            children = n.getField("children")
            if children:
                for i in range(children.getCount()):
                    queue.append(children.getMFNode(i))
        except:
            pass
    return None


def main():
    sup = Supervisor()
    timestep = int(sup.getBasicTimeStep())

    # Find robot
    robot_node = find_robot(sup)
    if not robot_node:
        raise RuntimeError(f"Robot not found (DEF='{ROBOT_DEF}', name='{ROBOT_NAME}')")

    # Get robot fields
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")

    # Get devices
    lidar = sup.getDevice("LDS-01")
    if not lidar:
        raise RuntimeError("LiDAR sensor 'LDS-01' not found")
    lidar.enable(timestep)
    lidar.enablePointCloud()

    left_motor = sup.getDevice("left wheel motor")
    right_motor = sup.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Load trajectory
    trajectory = load_trajectory(TRAJ_PATH)
    t0_traj = trajectory[0][0]
    t_end = trajectory[-1][0]
    traj_idx = 0

    print(f"[data_logger] Loaded trajectory: {len(trajectory)} waypoints, duration={t_end:.1f}s")
    print(f"[data_logger] Collecting data with dt={DT_LOG}s")

    # Setup output
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    f = open(out_path, 'w', newline='')
    writer = csv.writer(f)

    # CSV header
    header = ['timestamp']
    # LiDAR ranges (360 values)
    for i in range(360):
        header.append(f'range_{i}')
    # Odometry
    header.extend(['v', 'w'])
    # Ground truth
    header.extend(['gt_x', 'gt_y', 'gt_theta'])
    writer.writerow(header)

    next_log_time = 0.0
    trajectory_done = False

    print(f"[data_logger] Starting data collection...")

    while sup.step(timestep) != -1:
        sim_time = sup.getTime()

        # Only log at specified intervals
        if sim_time + 1e-9 < next_log_time:
            continue

        # Get commanded velocities from trajectory
        t_rel = sim_time + t0_traj

        # Advance trajectory index
        while traj_idx + 1 < len(trajectory) and trajectory[traj_idx + 1][0] <= t_rel:
            traj_idx += 1

        # Get current commanded velocities
        if t_rel < t_end:
            _, cmd_v, cmd_w = trajectory[traj_idx]
        else:
            cmd_v, cmd_w = 0.0, 0.0
            if not trajectory_done:
                print(f"[data_logger] Trajectory complete at t={sim_time:.2f}s")
                trajectory_done = True

        # Execute trajectory (convert to wheel velocities)
        vl = (2 * cmd_v - cmd_w * WHEEL_BASE) / (2 * WHEEL_RADIUS)
        vr = (2 * cmd_v + cmd_w * WHEEL_BASE) / (2 * WHEEL_RADIUS)
        left_motor.setVelocity(vl)
        right_motor.setVelocity(vr)

        # Read LiDAR
        ranges = lidar.getRangeImage()
        if len(ranges) != 360:
            print(f"[data_logger] Warning: Expected 360 ranges, got {len(ranges)}")
            continue

        # Get ground truth pose
        x, y, z = trans_field.getSFVec3f()
        ax, ay, az, ang = rot_field.getSFRotation()
        gt_theta = yaw_from_axis_angle(ax, ay, az, ang)

        # Write data row
        row = [f"{sim_time:.3f}"]
        # LiDAR ranges
        for r in ranges:
            row.append(f"{r:.6f}")
        # Odometry
        row.append(f"{cmd_v:.6f}")
        row.append(f"{cmd_w:.6f}")
        # Ground truth
        row.append(f"{x:.6f}")
        row.append(f"{y:.6f}")
        row.append(f"{gt_theta:.6f}")

        writer.writerow(row)

        next_log_time += DT_LOG

        # Stop if trajectory done and some time has passed
        if trajectory_done and sim_time > t_end + 1.0:
            break

    f.close()
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    print(f"[data_logger] Data collection complete!")
    print(f"[data_logger] Saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
