"""
Module: configs.py
Description:
    Configuration parameters for the 4-D UAV pathfinding simulation:
      - Window, camera, and FPS settings
      - Box dimensions and collision detection radii
      - UAV count and environment bounds
      - Grid resolution and iteration count
      - Statistics output path
"""

import pyray as pr
import numpy as np
# camera
window_w, window_h = 1280, 720

fps = 60
mapping_resolution = 1
#box dimentions
box_fixed_length = 10
buffer_zone_size = 10

# collision detection
collision_radius = buffer_zone_size
collision_spheres_dist = 1
intersection_collision_radius = 100


num_agents = 1

bounds = [(0, 8_000), (0, 8_000), (0, 150)] 
num_buildings,builduing_base_size =0, 15

# Calculate the center of the box
box_center = [np.mean(axis) for axis in bounds]  # [10.0, 10.0, 10.0]
camera = pr.Camera3D([bounds[0][1], bounds[1][1], bounds[2][1]], box_center, [0.0, 0.0, 1.0], 500.0, 0)

resolution = bounds[0][1]/10 # every x meters make a node in x and y axes
z_resolution = 25 # every x meters make a node in the z axis

num_iterations = 5 

# stats
#stats_storage_path = "stats/4D_UAV_pathfinding_stats.csv"

stats_storage_path = "stats/4D_UAV_pathfinding_stats_withObstacles.csv"


