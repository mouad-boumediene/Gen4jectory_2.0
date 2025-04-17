import pyray as pr
import numpy as np
# camera
window_w, window_h = 1280, 720

fps = 60
mapping_resolution = 1

seed = 21
#box dimentions
buffer_zone_size = 10
box_fixed_length = 10

# collision detection
collision_radius = buffer_zone_size
collision_spheres_dist = 1


num_agents = 25
bounds = [(0, 350), (0, 350), (0, 150)] 
num_buildings,builduing_base_size = 5, 20

# Calculate the center of the box
box_center = [np.mean(axis) for axis in bounds]  # [10.0, 10.0, 10.0]
camera = pr.Camera3D([bounds[0][1]*1.8, bounds[1][1]*1.8, bounds[2][1]*1.8], box_center, [0.0, 0.0, 1.0], 45.0, 0)

resolution = 50 # every x meters make a node in each axis

num_iterations = 1000 

# stats
stats_storage_path = "stats/4D_UAV_pathfinding_stats.csv"


