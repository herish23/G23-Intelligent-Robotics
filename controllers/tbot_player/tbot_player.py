from controller import Robot
import csv, os, sys

LEFT_MOTOR_NAME  = os.environ.get("LEFT_MOTOR_NAME",  "left wheel motor")
RIGHT_MOTOR_NAME = os.environ.get("RIGHT_MOTOR_NAME", "right wheel motor")

TRAJ = os.environ.get("TRAJ", "../../maps/traj/traj_debug_square.csv")
WHEEL_BASE   = float(os.environ.get("WHEEL_BASE", "0.160"))   # m
WHEEL_RADIUS = float(os.environ.get("WHEEL_RADIUS", "0.033")) # m

# If you also want Webots to auto-pause the whole sim once finished, set STOP_AND_PAUSE=1
STOP_AND_PAUSE = os.environ.get("STOP_AND_PAUSE", "0") == "1"

def read_traj(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = [(float(d['timestamp']),
                 float(d['target_linear_velocity']),
                 float(d['target_angular_velocity'])) for d in r]
    if not rows:
        raise RuntimeError(f"Trajectory file '{path}' is empty.")
    rows.sort(key=lambda x: x[0])
    return rows

def main():
    bot = Robot()
    ts = int(bot.getBasicTimeStep())

    try:
        lw = bot.getDevice(LEFT_MOTOR_NAME)
        rw = bot.getDevice(RIGHT_MOTOR_NAME)
    except Exception as e:
        print(f"[tbot_player] ERROR getting wheel motors: {e}", file=sys.stderr)
        return

    lw.setPosition(float('inf')); rw.setPosition(float('inf'))
    lw.setVelocity(0.0); rw.setVelocity(0.0)

    traj = read_traj(TRAJ)
    t0 = traj[0][0]
    t_end = traj[-1][0]
    idx = 0
    done = False

    print(f"[tbot_player] Loaded {len(traj)} points from {os.path.abspath(TRAJ)}")
    print(f"[tbot_player] Duration: {t_end - t0:.3f}s (one-shot). Will stop at end.")

    while bot.step(ts) != -1:
        t = bot.getTime()

        if done:
            # keep stopped
            lw.setVelocity(0.0); rw.setVelocity(0.0)
            continue

        # advance index while next timestamp has passed
        while idx + 1 < len(traj) and (traj[idx+1][0] - t0) <= t:
            idx += 1

        # if we've reached the final row (or time >= end), stop
        if idx >= len(traj) - 1 or t >= (t_end - t0):
            lw.setVelocity(0.0); rw.setVelocity(0.0)
            done = True
            print("[tbot_player] Trajectory complete. Wheels stopped.")
            if STOP_AND_PAUSE:
                # pause the whole simulation so logs stop growing
                try:
                    bot.simulationSetMode(Robot.SIMULATION_MODE_PAUSE)
                    print("[tbot_player] Simulation paused.")
                except Exception:
                    pass
            continue

        # follow current command
        _, v, w = traj[idx]
        vl = (2*v - w*WHEEL_BASE) / (2*WHEEL_RADIUS)
        vr = (2*v + w*WHEEL_BASE) / (2*WHEEL_RADIUS)
        lw.setVelocity(vl); rw.setVelocity(vr)

if __name__ == "__main__":
    main()
