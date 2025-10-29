## this is the first commit in the brnach amcl for the project
import sys
import random
import numpy as np

sys.path.append("../../libraries")
from localization_utils import load_sensor_data, load_map

## load sensor data and map
sensor_data = load_sensor_data('../../data/sensor_data_clean.csv')
map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
print(f"Loaded {len(sensor_data)} timesteps")
print(f"Map size: {map_info.width} x {map_info.height}")

## Particle class to initiate 
class Particle:
      def __init__(self, x, y, theta, weight=1.0):
          self.x = x           
          self.y = y           
          self.theta = theta   
          self.weight = weight 


## we create particles for the map 
## spread the particles around the map 
def initialize_particles(num_particles, map_info):
    particles = []

    ## map bound calculation for spreading the particles internal
    x_min = map_info.origin_x
    x_max = map_info.origin_x + (map_info.width * map_info.resolution)
    y_min = map_info.origin_y
    y_max = map_info.origin_y + (map_info.height * map_info.resolution)

    for i in range(num_particles):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        theta = random.uniform(-np.pi, np.pi)  
        w = 1.0 / num_particles  
        particles.append(Particle(x, y, theta, w))

    return particles

## function to normalise the weight off all the particles made from the top fx 
## we do this step to make sure the particles sum are 1
def normalize_weights(particles):
    total = sum(p.weight for p in particles)
    ## 500 particles = 1/500 
    if total > 0:
        for p in particles:
            p.weight /= total
    else:
        w = 1.0 / len(particles)
        for p in particles:
            p.weight = w


def get_mean_pose(particles):

    ## handle mean for <x,y> for the particle
    mean_x = sum(p.x * p.weight for p in particles)
    mean_y = sum(p.y * p.weight for p in particles)

    # wraparound (-pi to pi) for accurate calculations of the theta for the robto motion
    ## handle theta values for each particle
    sin_theta = sum(np.sin(p.theta) * p.weight for p in particles)
    cos_theta = sum(np.cos(p.theta) * p.weight for p in particles)
    mean_theta = np.arctan2(sin_theta, cos_theta)
    return mean_x, mean_y, mean_theta


## motion model - mapping 
## ($\alpha_1$, $\alpha_2$, $\alpha_3$, $\alpha_4$)
alpha1 = 0.2  # rot noise from rotational motion
alpha2 = 0.2  # rot noise from translation motion
alpha3 = 0.2  # translation noise from translation
alpha4 = 0.2  # translation noise from rotation

## fx to generate a random noise values from Gaussian
def sample_normal(b):
    std = np.sqrt(b)
    return np.random.normal(0, std)

## Prediction Step for motion model
def motion_update(particles, odom_prev, odom_curr):
    x_prev, y_prev, theta_prev = odom_prev
    x_curr, y_curr, theta_curr = odom_curr

    ## calculate changes in the odometry 
    delta_x = x_curr - x_prev
    delta_y = y_curr - y_prev

    # initial rotation to for the robto trvael 
    delta_rot1 = np.arctan2(delta_y, delta_x) - theta_prev
    delta_trans = np.sqrt(delta_x**2 + delta_y**2)

    # final rotation after movement
    delta_rot2 = theta_curr - theta_prev - delta_rot1

    ## normalize angles to [-pi, pi]
    delta_rot1 = np.arctan2(np.sin(delta_rot1), np.cos(delta_rot1))
    delta_rot2 = np.arctan2(np.sin(delta_rot2), np.cos(delta_rot2))

    # calculates the particle for each noisy particle
    for p in particles:
        ## initial rotation (orient)
        rot1_hat = delta_rot1 + sample_normal(alpha1 * delta_rot1**2 + alpha2 * delta_trans**2)

        ## translation (x to x1)
        trans_hat = delta_trans + sample_normal(alpha3 * delta_trans**2 + alpha4 * (delta_rot1**2 + delta_rot2**2))
        
        ## fina rotation (orient)
        rot2_hat = delta_rot2 + sample_normal(alpha1 * delta_rot2**2 + alpha2 * delta_trans**2)

        ## apply the noisy motion to particles 
        p.x += trans_hat * np.cos(p.theta + rot1_hat)
        p.y += trans_hat * np.sin(p.theta + rot1_hat)
        p.theta += rot1_hat + rot2_hat

        # normalize theta back to [-pi, pi]
        p.theta = np.arctan2(np.sin(p.theta), np.cos(p.theta))

