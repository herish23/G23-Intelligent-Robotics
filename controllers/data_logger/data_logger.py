## automatic data collection from webot based on the trajectory files


from controller import Supervisor
import csv
import math
import os

## config from env vars
ROBOT_DEF = os.environ.get("ROBOT_DEF", "TB3")
ROBOT_NAME = os.environ.get("ROBOT_NAME", "TurtleBot3Burger")
TRAJ_PATH = os.environ.get("TRAJ", "../../maps/traj/traj_balanced_rectangle.csv")
OUT_DIR = os.environ.get("OUT_DIR", "../../data")
OUT_FILE = os.environ.get("OUT_FILE", "sensor_data_clean.csv")
DT_LOG = float(os.environ.get("DT_LOG", "0.02"))  ## 20ms logging

WHEEL_BASE = 0.160  # meters between wheels
WHEEL_RADIUS = 0.033  # wheel radius in meters


def yaw_from_axis_angle(ax, ay, az, ang):
    ## convert axis-angle rotation to yaw (heading angle)
    ## webots uses axis-angle, we need yaw for 2d localization
    s = math.sin(ang / 2.0)
    c = math.cos(ang / 2.0)
    qx, qy, qz, qw = ax * s, ay * s, az * s, c
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def load_trajectory(path):
    ## load trajectory csv (timestamp, v, w)
    traj = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = float(r['timestamp'])
            v = float(r['target_linear_velocity'])
            w = float(r['target_angular_velocity'])
            traj.append((t, v, w))
    traj.sort(key=lambda x: x[0])  ## make sure sorted by time
    return traj


def find_robot(sup):
    ## find robot node in webots scene tree
    node = sup.getFromDef(ROBOT_DEF)
    if node:
        return node

    ## fallback - search by name if DEF didnt work
    root = sup.getRoot()
    queue = [root]
    while queue:
        n = queue.pop(0)
        try:
            name_field = n.getField("name")
            if name_field and name_field.getSFString() == ROBOT_NAME:
                return n
        except:
            pass  ## not all nodes have name field
        try:
            children = n.getField("children")
            if children:
                for i in range(children.getCount()):
                    queue.append(children.getMFNode(i))
        except:
            pass
    return None  ## didnt find robot


def main():
    sup = Supervisor()
    dt = int(sup.getBasicTimeStep())

    ## find robot in scene
    robot_node = find_robot(sup)
    if not robot_node:
        raise RuntimeError(f"cant find robot (DEF='{ROBOT_DEF}', name='{ROBOT_NAME}')")

    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")

    ## setup lidar
    lidar = sup.getDevice("LDS-01")
    if not lidar:
        raise RuntimeError("LDS-01 lidar not found")
    lidar.enable(dt)
    lidar.enablePointCloud()

    ## setup motors (velocity mode)
    left_motor = sup.getDevice("left wheel motor")
    right_motor = sup.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))  ## velocity control mode
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    ## enable wheel encoders for odometry computation
    left_enc = sup.getDevice("left wheel sensor")
    right_enc = sup.getDevice("right wheel sensor")
    left_enc.enable(dt)
    right_enc.enable(dt)

    ## load trajectory to follow
    traj = load_trajectory(TRAJ_PATH)
    t0 = traj[0][0]
    t_end = traj[-1][0]
    traj_idx = 0

    print(f"loaded trajectory: {len(traj)} waypoints, {t_end:.1f}s duration")
    print(f"logging at dt={DT_LOG}s")

    ## setup csv output
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    f = open(out_path, 'w', newline='')
    writer = csv.writer(f)

    
    header = ['timestamp']
    for i in range(360):
        header.append(f'range_{i}')
    header.extend(['odom_x', 'odom_y', 'odom_theta'])  ## odometry from wheel encoders
    header.extend(['v', 'w'])  ## commanded velocities (for debugging)
    header.extend(['gt_x', 'gt_y', 'gt_theta'])  ## ground truth from supervisor
    writer.writerow(header)

    next_log_t = 0.0
    traj_done = False

    ## init odometry from starting pose
    pos0 = trans_field.getSFVec3f()
    ax, ay, az, ang = rot_field.getSFRotation()
    theta0 = yaw_from_axis_angle(ax, ay, az, ang)

    odom_x = pos0[0]
    odom_y = pos0[1]
    odom_theta = theta0

    ## track encoder positions for delta computation
    prev_left = 0.0
    prev_right = 0.0
    first_step = True  ## skip odom update on first iter (no prev encoders yet)

    last_print = 0.0
    print_interval = 1.0  ## print status every second

    print("starting data collection...")
    print(f"start pos: ({odom_x:.3f}, {odom_y:.3f}, {math.degrees(odom_theta):.1f}°)")
    print(f"{'Time':>6s} | {'Cmd_v':>6s} {'Cmd_w':>6s} | {'GT_X':>8s} {'GT_Y':>8s} {'GT_θ':>7s} | {'Odom_X':>8s} {'Odom_Y':>8s} {'Odom_θ':>7s} | {'Err':>6s}")

    while sup.step(dt) != -1:
        t = sup.getTime()

        ## only log at specified intervals
        if t + 1e-9 < next_log_t:
            continue

        t_rel = t + t0

        ## find current trajectory waypoint
        while traj_idx + 1 < len(traj) and traj[traj_idx + 1][0] <= t_rel:
            traj_idx += 1

        ## get commanded velocities from trajectory
        if t_rel < t_end:
            _, cmd_v, cmd_w = traj[traj_idx]
        else:
            cmd_v, cmd_w = 0.0, 0.0
            if not traj_done:
                print(f"trajectory done at t={t:.2f}s")
                traj_done = True

        ## convert differential drive (v,w) to wheel velocities
        vl = (2 * cmd_v - cmd_w * WHEEL_BASE) / (2 * WHEEL_RADIUS)
        vr = (2 * cmd_v + cmd_w * WHEEL_BASE) / (2 * WHEEL_RADIUS)
        left_motor.setVelocity(vl)
        right_motor.setVelocity(vr)

        ## get lidar scan
        ranges = lidar.getRangeImage()
        if len(ranges) != 360:
            print(f"warning: got {len(ranges)} lidar beams instead of 360")
            continue

        ## get ground truth from supervisor
        x, y, z = trans_field.getSFVec3f()
        ax, ay, az, ang = rot_field.getSFRotation()
        gt_theta = yaw_from_axis_angle(ax, ay, az, ang)

        ## compute odometry from wheel encoders (differential drive)
        if not first_step:
            ## read encoder positions (in radians)
            left_pos = left_enc.getValue()
            right_pos = right_enc.getValue()

            ## compute wheel arc lengths since last step
            dl = (left_pos - prev_left) * WHEEL_RADIUS  ## meters
            dr = (right_pos - prev_right) * WHEEL_RADIUS

            ## differential drive kinematics
            ds = (dr + dl) / 2.0  ## linear displacement
            dtheta = (dr - dl) / WHEEL_BASE  ## angular displacement

            ## integrate motion into odometry pose
            odom_x += ds * math.cos(odom_theta + dtheta / 2.0)
            odom_y += ds * math.sin(odom_theta + dtheta / 2.0)
            odom_theta += dtheta
            odom_theta = math.atan2(math.sin(odom_theta), math.cos(odom_theta))  ## wrap to [-pi,pi]

            ## save for next step
            prev_left = left_pos
            prev_right = right_pos
        else:
            ## first timestep - just init encoders
            prev_left = left_enc.getValue()
            prev_right = right_enc.getValue()
            first_step = False

        ## print status every second
        if t - last_print >= print_interval:
            odom_err = math.sqrt((x - odom_x)**2 + (y - odom_y)**2)
            print(f"{t:6.2f} | {cmd_v:6.2f} {cmd_w:6.2f} | {x:8.3f} {y:8.3f} {math.degrees(gt_theta):7.1f} | {odom_x:8.3f} {odom_y:8.3f} {math.degrees(odom_theta):7.1f} | {odom_err*100:5.1f}cm")
            last_print = t

        ## write csv row
        row = [f"{t:.3f}"]
        for r in ranges:
            row.append(f"{r:.6f}")
        row.append(f"{odom_x:.6f}")
        row.append(f"{odom_y:.6f}")
        row.append(f"{odom_theta:.6f}")
        row.append(f"{cmd_v:.6f}")
        row.append(f"{cmd_w:.6f}")
        row.append(f"{x:.6f}")
        row.append(f"{y:.6f}")
        row.append(f"{gt_theta:.6f}")
        writer.writerow(row)

        next_log_t += DT_LOG

        if traj_done and t > t_end + 1.0:
            break

    f.close()
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    ## final odometry drift check
    x, y, z = trans_field.getSFVec3f()
    ax, ay, az, ang = rot_field.getSFRotation()
    gt_theta = yaw_from_axis_angle(ax, ay, az, ang)
    final_err = math.sqrt((x - odom_x)**2 + (y - odom_y)**2)
    theta_err = math.degrees(abs(gt_theta - odom_theta))
    if theta_err > 180:
        theta_err = 360 - theta_err

    print("\n" + "="*60)
    print("odometry drift summary")
    print("="*60)
    print(f"ground truth: ({x:.3f}, {y:.3f}, {math.degrees(gt_theta):.1f}°)")
    print(f"odometry:     ({odom_x:.3f}, {odom_y:.3f}, {math.degrees(odom_theta):.1f}°)")
    print(f"drift: {final_err*100:.1f}cm position, {theta_err:.1f}° heading")
    print("="*60)

    print("\ndata collection done!")
    print(f"saved: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
