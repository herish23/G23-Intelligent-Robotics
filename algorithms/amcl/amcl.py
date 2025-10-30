## this is the first commit in the brnach amcl for the project
import sys
import random
import numpy as np

sys.path.append("../../libraries")
from localization_utils import load_sensor_data, load_map, compute_likelihood_field

## load sensor data and map
sensor_data = load_sensor_data('../../data/sensor_data_clean.csv')
map_info = load_map('../../maps/epuck_world_map.pgm', '../../maps/epuck_world_map.yaml')
print(f"Loaded {len(sensor_data)} timesteps")
print(f"Map size: {map_info.width} x {map_info.height}")

## compute likelihood field
likelihood_field = compute_likelihood_field(map_info, max_dist=2.0)
print("Likelihood field ready")

## Particle class to initiate 
class Particle:
      def __init__(self, x, y, theta, weight=1.0):
          self.x = x           
          self.y = y           
          self.theta = theta   
          self.weight = weight 


## spread particles across map bounds
def initialize_particles(num_particles, map_info):
    particles = []

    ## map boundaries
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

## normalize particle weights
def normalize_weights(particles):
    total = sum(p.weight for p in particles)
    if total > 0:
        for p in particles:
            p.weight /= total
    else:
        # all zero weights - reset to uniform
        w = 1.0 / len(particles)
        for p in particles:
            p.weight = w


def get_mean_pose(particles):
    mean_x = sum(p.x * p.weight for p in particles)
    mean_y = sum(p.y * p.weight for p in particles)

    # circular mean for theta (handles wraparound)
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
# TODO: might need to tune these based on test results

## sample from gaussian noise
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


## Sensor model 

## sensor params for matching lidar to map
z_hit = 0.95  # probabliity for accurate measuremtns
z_rand = 0.05  # random noise value
sigma_hit = 0.2  # std deviation for the gaussian noises

## lidar max from turtlebot
max_range = 3.5

## we use 60 only from 360 that we have logged in csv
num_beams = 60
# check : using 60 for speed, might need more for accuracy 

## Correction step - weight particles using lidar scan matching
def sensor_update(particles, lidar_ranges, likelihood_field, map_info):
    angle_inc = 2 * np.pi / 360
    beam_step = 360 // num_beams

    for p in particles:
        w = 1.0

        ## loop through subset of beams
        for i in range(0, 360, beam_step):
            z = lidar_ranges[i]

            # skip bad readings
            if z >= max_range or np.isnan(z):
                continue

            ## where does this beam hit in world coords
            angle = p.theta + (i * angle_inc)
            hit_x = p.x + z * np.cos(angle)
            hit_y = p.y + z * np.sin(angle)

            # convert to map grid cells
            mx = int((hit_x - map_info.origin_x) / map_info.resolution)
            my = int((hit_y - map_info.origin_y) / map_info.resolution)

            # inside map?
            if 0 <= mx < map_info.width and 0 <= my < map_info.height:
                ## lookup distance to nearest obsacle
                d = likelihood_field[my, mx]

                # gaussian prob - small distance = high weight
                prob_hit = np.exp(-(d**2) / (2 * sigma_hit**2))

                # mix of gaussian + uniform random
                prob_z = z_hit * prob_hit + z_rand / max_range
                w *= prob_z
            else:
                # outside map, penalize heavily
                w *= 0.01  # check: is this too harsh?

        p.weight = w


## low variance sampling to reduce sampling error
def resample(particles):
    n = len(particles)

    # normalize weights first
    w_sum = sum(p.weight for p in particles)
    if w_sum == 0:
        # all weights zero, return uniform
        for p in particles:
            p.weight = 1.0 / n
        return particles

    for p in particles:
        p.weight /= w_sum

    # low variance resampling
    new_p = []
    r = random.uniform(0, 1.0/n)  # random start
    c = particles[0].weight
    i = 0
    # CHECK : if this needs optimization for large particle sets

    for m in range(n):
        u = r + m / n
        while u > c:
            i += 1
            c += particles[i].weight
        # copy particle
        new_p.append(Particle(particles[i].x, particles[i].y, particles[i].theta))

    # reset weights to uniform
    for p in new_p:
        p.weight = 1.0 / n

    return new_p


particles = initialize_particles(100, map_info)
print("particles", len(particles))
print("first particle ", particles[0].x, particles[0].y)
# motion test
gt0 = sensor_data['ground_truth'][0]
gt1 = sensor_data['ground_truth'][1]
prev_odom = (gt0[0], gt0[1], gt0[2])
curr_odom = (gt1[0], gt1[1], gt1[2])
motion_update(particles, prev_odom, curr_odom)
print("after motion", particles[0].x, particles[0].y)
# sensor update
lidar_data = sensor_data['lidar_scans'][1]
sensor_update(particles, lidar_data, likelihood_field, map_info)
print("weights:", particles[0].weight, particles[1].weight)
normalize_weights(particles)

# try resample
particles = resample(particles)
print("resampled:", len(particles))
mx, my, mtheta = get_mean_pose(particles)
print(f"est {mx:.3f}, {my:.3f}, {mtheta:.3f}")
print(f"gt  {gt1[0]:.3f}, {gt1[1]:.3f}, {gt1[2]:.3f}")
err = np.sqrt((mx-gt1[0])**2 + (my-gt1[1])**2)
print(f"error: {err:.3f}m")




## KLD adaptive sampling params
n_min = 100
n_max = 5000
epsilon = 0.05  # KLD error bound
z_quantile = 2.58  # chi-square (99% confidence)
bin_size = 0.5  # 50cm bins for x,y


# test and change according to the output 

## sequential KLD resampling 
def kld_resample(particles, map_info):
    n = len(particles)

    # normalize weights
    w_sum = sum(p.weight for p in particles)
    if w_sum == 0:
        for p in particles:
            p.weight = 1.0 / n
        return particles

    for p in particles:
        p.weight /= w_sum

    # build cumulative weights for sampling
    cum_w = []
    c = 0.0
    for p in particles:
        c += p.weight
        cum_w.append(c)

    #sample one at a time until KLD bound satisfied
    new_p = []
    bins = {}
    k = 0  # bins with support
    M_x = 0  # particle count

    while True:
        # compute threshold from KLD bound
        if k > 1:

            # KLD bound calculation from chi-square
            thresh = (k - 1) / (2 * epsilon) * (1 - 2/(9*(k-1)) + np.sqrt(2/(9*(k-1))) * z_quantile)**3
            if M_x >= thresh and M_x >= n_min:
                break

        if M_x >= n_max:
            break

        # sample one particle independently
        r = random.uniform(0, 1.0)
        idx = 0
        for i in range(n):
            if cum_w[i] >= r:
                idx = i
                break

        # add particle
        p_new = Particle(particles[idx].x, particles[idx].y, particles[idx].theta)
        new_p.append(p_new)
        M_x += 1

        # check if new bin (map-relative binning)
        bx = int(np.floor((p_new.x - map_info.origin_x) / bin_size))
        by = int(np.floor((p_new.y - map_info.origin_y) / bin_size))
        theta_wrap = p_new.theta + np.pi  # shift [-pi,pi] to [0,2pi]
        btheta = int(np.floor(theta_wrap / (np.pi/4))) % 8  # 8 bins, wrap around
        key = (bx, by, btheta)
        if key not in bins:
            k += 1
            bins[key] = True

    # uniform weights after resampling
    for p in new_p:
        p.weight = 1.0 / len(new_p)

    return new_p


p_spread = initialize_particles(500, map_info)
p_spread_kld = kld_resample(p_spread, map_info)
print(f"spread {len(p_spread)}, {len(p_spread_kld)}")

p_conv = []
for i in range(500):
    x = -0.5 + random.uniform(-0.1, 0.1)
    y = -0.5 + random.uniform(-0.1, 0.1)
    theta = random.uniform(-0.2, 0.2)
    p_conv.append(Particle(x, y, theta, 1.0/500))

p_conv_kld = kld_resample(p_conv, map_info)
print(f"converged {len(p_conv)} , {len(p_conv_kld)}")


