# Comparative Evaluation of EKF, Markov Grid, and Adaptive Monte Carlo (AMCL with KLD-Sampling) Localization for Mobile Robots

---

## Running the Simulation

To start:

1. Open Webots
2. Load the world file: `worlds/exp.wbt`

The world has a TurtleBot3 with LiDAR on a checkboard arena with 5 wooden box obstacles.

---

## Data Collection

### How it Works

The `data_logger` controller runs the robot through a predefined trajectory and collects sensor data for offline processing. It logs:

* LiDAR range readings (360 points)
* Commanded velocities (v, w)
* Ground truth position (x, y, theta) from Webots supervisor

All data is saved to:

```
data/sensor_data_clean.csv
```
## Trajectories

Trajectory files are in `maps/traj/`:

* **traj_debug_square.csv** — Simple 0.20m x 0.20m square path (default). Good for initial testing.
* **traj_eval_complex.csv** — Longer, more complex path for full evaluation.

You can change which trajectory is used by editing the `TRAJ_PATH` variable in `controllers/data_logger/data_logger.py` or set the `TRAJ` environment variable.

---

## Map Files

The map files needed for localization are in `maps/`:

* `epuck_world_map.pgm` — occupancy grid map
* `epuck_world_map.yaml` — map metadata (resolution, origin, etc)

These were generated from the Webots world and match the 5 box obstacles in the simulation.

**Note:** If you change the world layout, you'll need to regenerate these files.

---

## Configuration Files

Multiple config files are provided in `configs/` to test different conditions:

* `config_baseline.yaml` — no extra noise
* `config_noise_XX.yaml` — adds 10%-80% noise to sensors
* `config_init_X.yaml` — tests recovery from wrong initial pose
* `config_sparse_4.yaml` — uses only 1/4 of LiDAR points

These let you test how robust each algorithm is under different conditions.

---

## Offline Processing

After collecting data, you'll run the three localization algorithms (EKF, Markov Grid, AMCL) on the saved CSV data. Each algorithm will:

1. Load the sensor data
2. Process odometry + LiDAR readings
3. Estimate robot position at each timestep
4. Compare against ground truth to calculate error metrics

This setup lets us do fair comparisons since all algorithms process the exact same data.

---



