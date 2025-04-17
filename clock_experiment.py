from src.planners import ThetaStar
from src.agent import UAVGenerator, UAV
from src.utils import gen_endbox, gen_startbox  # Make sure this import is correct
from src.utils import get_LoS

import numpy as np
import logging
from src.visualization_raylib import animate_raylib
from tqdm import tqdm
import random

np.random.seed(1)
random.seed(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

#np.random.seed(configs.seed)

# Map Creation
logging.info("Creating a map ...")
bounds = [(0, 350), (0, 350), (0, 150)] 

# Generating the Drones
logging.info("Generating the Drones ...")
uav_generator = UAVGenerator()
uavs:list[UAV] = [uav_generator.gen_uav(drone_id=id) for id in range(12)]


# Generating start and goal positions
logging.info("Generating starts and goals ...")
# Initializing the Theta star
theta_star = ThetaStar([],bounds, resolution=25)


end_nodes = uav_generator.gen_clock_endpoints(theta_star.graph, radius=150, center=(150,150),num_neighbors=16)


theta_star.visualize_graph_3d()

# generate end boxes
for i, uav in enumerate(uavs):
    start, goal = end_nodes[i]
    start_position = theta_star.graph.nodes[start]["coords"]
    goal_position = theta_star.graph.nodes[goal]["coords"]
    uav.end_hitbox = gen_endbox(start_position,goal_position, uav.max_velocity, uav.drone_id,dynamic_time_span=True)
    uav.start_hitbox = gen_startbox(start_position,goal_position, uav.max_velocity, uav.drone_id, dynamic_time_span=True)
    theta_star.hit_boxes.extend([[uav.end_hitbox, uav.start_hitbox]])

# Path planning and smoothing
logging.info("Path planning and smoothing ...")
for i, uav in enumerate(tqdm(uavs, desc="Planning paths")):
    start, goal = end_nodes[i]
    start_position = theta_star.graph.nodes[start]["coords"]
    uav.plan_path(uavs,planner=theta_star, start=start, goal=goal)
    if uav.path is None:
        uav.start_hitbox = gen_startbox(start_position,drone_id=uav.drone_id)
        uav.hit_boxes = [[uav.start_hitbox]]

LoS,LoS_UAVs = get_LoS(uavs)

# Visualization
logging.info("Visualization ...")

#visualization.plot_uav_paths(uavs, obstacles, graph=theta_star.graph, end_points = end_nodes)
animate_raylib(uavs, [], graph=theta_star.graph, bounds=bounds, end_nodes = end_nodes, show_only = LoS_UAVs, clock_scen = True)
