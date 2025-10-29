## this is the first commit in the brnach amcl for the project
import sys
import pandas as pd
import random
import numpy as np

sys.path.append("../../libraries")
from localization_utils import load_sensor_data, load_map

df = pd.read_csv('../../data/sensor_data_clean.csv')
map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
print(f"Map size: {map_info.width} x {map_info.height}")
print(df.head())

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

