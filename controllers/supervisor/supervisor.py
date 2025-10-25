from controller import Supervisor
import csv, math, os, time

# -------- config --------
ROBOT_DEF      = os.environ.get("ROBOT_DEF", "TB3")                  # set to your robot DEF if you added one
ROBOT_NAME     = os.environ.get("ROBOT_NAME", "TurtleBot3Burger")    # fallback: search by name
OUT_DIR        = os.environ.get("OUT_DIR", "../../maps/logs")
LOG_BASENAME   = os.environ.get("LOG_BASENAME", "run")
DT_LOG         = float(os.environ.get("DT_LOG", "0.02"))             # log step (s)
TRAJ_PATH      = os.environ.get("TRAJ", "../../maps/traj/traj_debug_square.csv")
ANCHOR_TO_GT0  = os.environ.get("ANCHOR_TO_GT0", "1") == "1"         # start ideal path at first GT pose

# -------- helpers --------
def yaw_from_axis_angle(ax, ay, az, ang):
    s = math.sin(ang/2.0); c = math.cos(ang/2.0)
    qx, qy, qz, qw = ax*s, ay*s, az*s, c
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def load_traj(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = [(float(d['timestamp']),
                 float(d['target_linear_velocity']),
                 float(d['target_angular_velocity'])) for d in r]
    if not rows:
        raise RuntimeError(f"Trajectory file '{path}' is empty.")
    rows.sort(key=lambda x: x[0])
    return rows

def find_robot(sup: Supervisor):
    # Prefer DEF
    node = sup.getFromDef(ROBOT_DEF)
    if node is not None:
        return node
    # Fallback: BFS by "name" field
    root = sup.getRoot()
    q = [root]
    while q:
        n = q.pop(0)
        try:
            name_field = n.getField("name")
            if name_field and name_field.getSFString() == ROBOT_NAME:
                return n
        except Exception:
            pass
        try:
            ch = n.getField("children")
            if ch:
                for i in range(ch.getCount()):
                    q.append(ch.getMFNode(i))
        except Exception:
            pass
    return None

# -------- main --------
def main():
    sup = Supervisor()
    ts = int(sup.getBasicTimeStep())

    robot = find_robot(sup)
    if robot is None:
        raise RuntimeError(f"Supervisor: could not find robot by DEF '{ROBOT_DEF}' or name '{ROBOT_NAME}'.")

    trans = robot.getField("translation")
    rot   = robot.getField("rotation")

    traj = load_traj(TRAJ_PATH)
    t0_traj = traj[0][0]
    t1_traj = traj[-1][0]

    # Prepare output
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{LOG_BASENAME}_gt.csv")  # overwrite each run (clean run file)
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "sim_time_s",
        "cmd_v_mps","cmd_w_rps",
        "ideal_x_m","ideal_y_m","ideal_yaw_rad",
        "gt_x_m","gt_y_m","gt_z_m","gt_yaw_rad"
    ])

    # Ideal integration state
    have_anchor = False
    ideal_x = ideal_y = ideal_yaw = 0.0

    # Traj index for fast lookup
    idx = 0

    next_log_t = 0.0
    last_sim_t = 0.0

    while sup.step(ts) != -1:
        sim_t = sup.getTime()

        # log at DT_LOG
        if sim_t + 1e-9 < next_log_t:
            continue

        # ----- ground truth -----
        x, y, z = trans.getSFVec3f()
        ax, ay, az, ang = rot.getSFRotation()
        gt_yaw = yaw_from_axis_angle(ax, ay, az, ang)

        # anchor ideal start to first GT pose (nice for overlay)
        if not have_anchor and ANCHOR_TO_GT0:
            ideal_x, ideal_y, ideal_yaw = x, y, gt_yaw
            have_anchor = True

        # ----- commanded v,w at current time (relative to traj start) -----
        t_rel = (sim_t) + t0_traj  # we play from t=0 → use traj timestamps as-is
        # advance idx to the last timestamp <= t_rel (hold-last behavior)
        while idx + 1 < len(traj) and traj[idx + 1][0] <= t_rel:
            idx += 1
        cmd_v, cmd_w = traj[idx][1], traj[idx][2]

        # If beyond end of trajectory, hold zeros
        if t_rel >= t1_traj:
            cmd_v, cmd_w = 0.0, 0.0

        # ----- integrate ideal pose (unicycle) using the log step -----
        dt = max(1e-9, sim_t - last_sim_t) if last_sim_t > 0 else DT_LOG
        ideal_x += cmd_v * math.cos(ideal_yaw) * dt
        ideal_y += cmd_v * math.sin(ideal_yaw) * dt
        ideal_yaw += cmd_w * dt

        # write row
        w.writerow([
            f"{sim_t:.3f}",
            f"{cmd_v:.6f}", f"{cmd_w:.6f}",
            f"{ideal_x:.6f}", f"{ideal_y:.6f}", f"{ideal_yaw:.6f}",
            f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{gt_yaw:.6f}"
        ])

        last_sim_t = sim_t
        next_log_t += DT_LOG

    f.close()

if __name__ == "__main__":
    main()
