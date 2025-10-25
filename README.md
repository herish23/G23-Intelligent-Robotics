# Experiment of Extended Kalman Filter (EKF), Markov Grid Localisation, and Adaptive Monte Carlo Localisation (AMCL with KLD-Sampling)

This is the first commit following the successful submission and approval of our proposal.

---

## Using the Preset Map and Robot

To start the simulation:

1. Open Webots.
2. Load the world file located at:

   ```
   worlds/exp.wbt
   ```

This world contains:

* A TurtleBot equipped with a LiDAR sensor.
* A Supervisor node that records ground-truth data and saves it to CSV logs.

---

## World and Data Files

### Logging

The supervisor automatically logs ground-truth values to:

```
maps/logs/run_gt.csv
```

### Map Files

The `.pgm` and `.yaml` files required for localisation are stored in:

```
maps/
```

Note: If you make any changes to the map, even small ones, you must generate new `.pgm` and `.yaml` files, since they are derived directly from the 3D world.

### Trajectories

Sample trajectories are included in:

```
maps/traj/
```

The following two example files are provided:

* `traj_debug_square.csv` — default trajectory loaded at startup.
* `traj_eval_complex.csv` — a more complex evaluation trajectory.

Ground-truth (GT) values in the logs correspond to the `traj_debug_square.csv` path.

---

## Controllers

Two controllers are included in this project:

1. **Supervisor** — Records motion data and logs ground-truth positions.
2. **Player** — Controls the TurtleBot equipped with LiDAR, used to implement and test the localisation algorithms (EKF, Markov Grid, AMCL).

---

## Notes

* Any modification to the world or map will require regenerating the corresponding map files (`.pgm` and `.yaml`).
* The default configuration is set up for initial experimentation and algorithm testing.

---

