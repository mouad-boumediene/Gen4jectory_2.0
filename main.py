"""
Module: main_simulation.py
Description:
    Main entry-point script for 4-D trajectory planning:
      - Generates a 3D city map (buildings).
      - Instantiates UAVs and sampling start/goal hit-boxes.
      - Plans UAV paths using the Theta* planner.
      - Computes loss-of-separation (LoS) metrics.
      - Visualizes results via Raylib animation.
"""

from src.planners import ThetaStar
from src.agent import UAVGenerator, UAV
from src.utils import gen_endbox, gen_startbox  # Make sure this import is correct
from src.utils import get_LoS, generate_buildings, compute_segment_intersections, generate_city

import numpy as np
from src import configs
import logging
from src.visualization_raylib import animate_raylib
from tqdm import tqdm
import random
import time
np.random.seed(124)
random.seed(124)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


#np.random.seed(configs.seed)

# Map Creation
logging.info("Creating a map ...")
bounds = configs.bounds
buildings = []  # Initialize buildings as an empty list
#buildings = generate_buildings(num_buildings=configs.num_buildings, base_size=configs.builduing_base_size)
buildings = generate_city(bounds, obstacle_density=0.0175, base_size_pct=0.025, min_height=5, max_height=configs.bounds[2][1]-configs.bounds[2][0])
# Generating the Drones
logging.info("Generating the Drones ...")
uav_generator = UAVGenerator()
uavs:list[UAV] = [uav_generator.gen_uav(drone_id=i) for i in range(configs.num_agents)]


# Generating start and goal positions
logging.info("Generating starts and goals ...")
#end_nodes = uav_generator.gen_fixed_endpoints(z=10)
# Initializing the Theta star
theta_star = ThetaStar(buildings,bounds, resolution=configs.resolution)


theta_star.visualize_graph_3d()


end_nodes = uav_generator.gen_random_endpoints(theta_star.graph, bounds, configs.num_agents, obstacles=buildings)

# Compute intersections on the finite segments
intersections = compute_segment_intersections(theta_star.graph.nodes, end_nodes)
""" print("Intersections per UAV:")
for i, lst in enumerate(intersections):
    print(f" UAV {i}:", lst)
exit() """

#end_nodes = uav_generator.gen_clock_endpoints(theta_star.graph)


theta_star.visualize_graph_3d()

# generate end boxes
for i, uav in enumerate(uavs):
    start, goal = end_nodes[i]
    start_position = theta_star.graph.nodes[start]["coords"]
    goal_position = theta_star.graph.nodes[goal]["coords"]
    uav.end_hitbox = gen_endbox(start_position,goal_position, uav.max_velocity, uav.drone_id,dynamic_time_span=False)
    uav.start_hitbox = gen_startbox(start_position,goal_position, uav.max_velocity, uav.drone_id,dynamic_time_span=False)
    uav.hit_boxes = [[uav.end_hitbox, uav.start_hitbox]]
    theta_star.hit_boxes.extend([[uav.end_hitbox, uav.start_hitbox]])
#theta_star.visualize_graph_3d()


#end_nodes = uav_generator.generate_intersection_scene(theta_star.graph,configs.num_agents)

# Path planning and smoothing
logging.info("Path planning ...")

start_time_point = time.time()  # Start time measurement
for i, uav in enumerate(tqdm(uavs, desc="Planning paths")):
    start, goal = end_nodes[i]
    start_position = theta_star.graph.nodes[start]["coords"]
    direct_path_intersections = intersections[i]
    uav.plan_path(uavs, planner=theta_star, start=start, goal=goal, endpoints_intersections=direct_path_intersections)
    
    if uav.path is None:
        print('no path is found')
        uav.start_hitbox = gen_startbox(start_position,drone_id=uav.drone_id)
        uav.hit_boxes = [[uav.start_hitbox]]

end_time_point = time.time()  # End time measurement
    
# Calculate the elapsed time
elapsed_time = end_time_point - start_time_point

# Print the time taken
print(f"Planning took {elapsed_time:.4f} seconds")
logging.info("Path planning finished !")


LoS,LoS_UAVs = get_LoS(uavs)
#print("LoS of this Simulation is : ", get_LoS(uavs))
# Visualization
logging.info("Visualization ...")

#visualization.plot_uav_paths(uavs, obstacles, graph=theta_star.graph, end_points = end_nodes)
animate_raylib(uavs, buildings, graph=theta_star.graph, end_nodes = end_nodes, bounds=configs.bounds, show_only = LoS_UAVs, draw_endboxes = True, intersections= intersections)
