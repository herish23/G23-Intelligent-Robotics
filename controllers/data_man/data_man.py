## this teleop file to collec data to sensor_data_clean.csv 

from controller import Supervisor, Keyboard
import csv
import math
import os

## config
OUT_DIR = "../../data"
OUT_FILE = "sensor_data_clean.csv"
DT_LOG = 0.02  ## 20ms logging

## turtlebot3 specs
WHEEL_BASE = 0.160
WHEEL_RADIUS = 0.033

def yaw_from_axis_angle(ax, ay, az, ang):
    ## convert axis-angle to yaw
    s = math.sin(ang / 2.0)
    c = math.cos(ang / 2.0)
    qx, qy, qz, qw = ax * s, ay * s, az * s, c
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def find_robot(sup):
    ## find robot node
    node = sup.getFromDef("TB3")
    if node:
        return node
    root = sup.getRoot()
    queue = [root]
    while queue:
        n = queue.pop(0)
        try:
            name_field = n.getField("name")
            if name_field and name_field.getSFString() == "TurtleBot3Burger":
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
    dt = int(sup.getBasicTimeStep())

    ## find robot
    robot_node = find_robot(sup)
    if not robot_node:
        raise RuntimeError("cant find robot")

    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")

    ## setup lidar
    lidar = sup.getDevice("LDS-01")
    if not lidar:
        raise RuntimeError("LDS-01 not found")
    lidar.enable(dt)
    lidar.enablePointCloud()

    ## setup motors
    left_motor = sup.getDevice("left wheel motor")
    right_motor = sup.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    ## setup wheel encoders
    left_enc = sup.getDevice("left wheel sensor")
    right_enc = sup.getDevice("right wheel sensor")
    left_enc.enable(dt)
    right_enc.enable(dt)

    ## setup keyboard
    keyboard = sup.getKeyboard()
    keyboard.enable(dt)

    ## setup csv output
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILE)
    f = open(out_path, 'w', newline='')
    writer = csv.writer(f)

    ## csv header
    header = ['timestamp']
    for i in range(360):
        header.append(f'range_{i}')
    header.extend(['odom_x', 'odom_y', 'odom_theta'])
    header.extend(['v', 'w'])
    header.extend(['gt_x', 'gt_y', 'gt_theta'])
    writer.writerow(header)

    ## init odometry
    pos0 = trans_field.getSFVec3f()
    ax, ay, az, ang = rot_field.getSFRotation()
    theta0 = yaw_from_axis_angle(ax, ay, az, ang)

    odom_x = pos0[0]
    odom_y = pos0[1]
    odom_theta = theta0

    prev_left = 0.0
    prev_right = 0.0
    first_step = True

    next_log_t = 0.0
    last_print = 0.0

    ## keyboard control params
    speed = 0.15  ## linear speed
    turn_speed = 1.0  ## angular speed

    print("="*60)
    print("MANUAL DATA LOGGER - KEYBOARD CONTROL")
    print("="*60)
    print("Controls:")
    print("  UP arrow    : Forward")
    print("  DOWN arrow  : Backward")
    print("  LEFT arrow  : Turn left")
    print("  RIGHT arrow : Turn right")
    print("  SPACE       : Stop")
    print("="*60)
    print("Drive around the map avoiding obstacles")
    print("Press STOP button in Webots when done")
    print("="*60)
    print(f"start pos: ({odom_x:.3f}, {odom_y:.3f}, {math.degrees(odom_theta):.1f}°)")
    print(f"{'Time':>6s} | {'Cmd_v':>6s} {'Cmd_w':>6s} | {'GT_X':>8s} {'GT_Y':>8s} | {'Odom_X':>8s} {'Odom_Y':>8s} | {'Err':>6s}")

    while sup.step(dt) != -1:
        t = sup.getTime()

        ## keyboard control
        key = keyboard.getKey()
        cmd_v = 0.0
        cmd_w = 0.0

        if key == keyboard.UP:
            cmd_v = speed
        elif key == keyboard.DOWN:
            cmd_v = -speed
        elif key == keyboard.LEFT:
            cmd_w = turn_speed
        elif key == keyboard.RIGHT:
            cmd_w = -turn_speed
        elif key == ord(' '):
            cmd_v = 0.0
            cmd_w = 0.0

        ## set motor velocities
        vl = (2 * cmd_v - cmd_w * WHEEL_BASE) / (2 * WHEEL_RADIUS)
        vr = (2 * cmd_v + cmd_w * WHEEL_BASE) / (2 * WHEEL_RADIUS)
        left_motor.setVelocity(vl)
        right_motor.setVelocity(vr)

        ## only log at intervals
        if t + 1e-9 < next_log_t:
            continue

        ## get lidar
        ranges = lidar.getRangeImage()
        if len(ranges) != 360:
            continue

        ## get ground truth
        x, y, z = trans_field.getSFVec3f()
        ax, ay, az, ang = rot_field.getSFRotation()
        gt_theta = yaw_from_axis_angle(ax, ay, az, ang)

        ## compute odometry from encoders
        if not first_step:
            left_pos = left_enc.getValue()
            right_pos = right_enc.getValue()

            dl = (left_pos - prev_left) * WHEEL_RADIUS
            dr = (right_pos - prev_right) * WHEEL_RADIUS

            ds = (dr + dl) / 2.0
            dtheta = (dr - dl) / WHEEL_BASE

            odom_x += ds * math.cos(odom_theta + dtheta / 2.0)
            odom_y += ds * math.sin(odom_theta + dtheta / 2.0)
            odom_theta += dtheta
            odom_theta = math.atan2(math.sin(odom_theta), math.cos(odom_theta))

            prev_left = left_pos
            prev_right = right_pos
        else:
            prev_left = left_enc.getValue()
            prev_right = right_enc.getValue()
            first_step = False

        ## print status
        if t - last_print >= 1.0:
            odom_err = math.sqrt((x - odom_x)**2 + (y - odom_y)**2)
            print(f"{t:6.2f} | {cmd_v:6.2f} {cmd_w:6.2f} | {x:8.3f} {y:8.3f} | {odom_x:8.3f} {odom_y:8.3f} | {odom_err*100:5.1f}cm")
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

    f.close()
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    ## final odom drift
    x, y, z = trans_field.getSFVec3f()
    ax, ay, az, ang = rot_field.getSFRotation()
    gt_theta = yaw_from_axis_angle(ax, ay, az, ang)
    final_err = math.sqrt((x - odom_x)**2 + (y - odom_y)**2)

    print("MANUAL RECORDING COMPLETE!")


if __name__ == "__main__":
    main()
