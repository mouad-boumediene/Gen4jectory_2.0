from src.planners import ThetaStar
from src.agent import UAVGenerator, UAV
from src.utils import gen_endbox, gen_startbox  # Make sure this import is correct
from src.utils import get_LoS, generate_buildings

import numpy as np
from src import configs
import logging
from src.visualization_raylib import animate_raylib
from tqdm import tqdm
import random
import time
import csv 

def store_results(dictionary, filename, print_line=False):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        
        # Print the header and the line being written if print_line is True
        if print_line:
            print("Header:", ', '.join(dictionary.keys()))
            print("Line:  ", ', '.join(map(str, dictionary.values())))
        
        writer.writerow(dictionary)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


# Number of iterations to run the script
num_iterations = 10  # Set this to however many times you want the script to run
visualize = False
for scen in range(num_iterations):
    np.random.seed(scen)
    random.seed(scen)
    logging.info(f"scenario {scen + 1}/{num_iterations}")


    TEST_STATS = {
    'sim_num': scen,
    'num_agents': configs.num_agents,
    'solver': 'Theta*',
    'LoS':False,
    'solved': True,
    'comp_time': 0.0,
    'ftime':np.inf,
    'simple_ftime':np.inf,
    'map_size':(configs.bounds[0][1], configs.bounds[1][0], configs.bounds[2][1]) ,
    'resolution':configs.resolution,
    'num_buildings':configs.num_buildings,
    'building_base_size': configs.builduing_base_size}

    # Map Creation
    logging.info("Creating a map ...")
    bounds = configs.bounds
    buildings = generate_buildings(num_buildings=configs.num_buildings, base_size=configs.builduing_base_size)

    # Generating the Drones
    logging.info("Generating the Drones ...")
    uav_generator = UAVGenerator()
    uavs: list[UAV] = [uav_generator.gen_uav(drone_id=i) for i in range(configs.num_agents)]

    # Generating start and goal positions
    logging.info("Generating starts and goals ...")
    theta_star = ThetaStar(buildings, bounds, resolution=configs.resolution)

    #theta_star.visualize_graph_3d()

    end_nodes = uav_generator.gen_random_endpoints(theta_star.graph, bounds, configs.num_agents, obstacles=buildings)
    #theta_star.visualize_graph_3d()

    # Generate end boxes
    for i, uav in enumerate(uavs):
        start, goal = end_nodes[i]
        start_position = theta_star.graph.nodes[start]["coords"]
        goal_position = theta_star.graph.nodes[goal]["coords"]
        uav.end_hitbox = gen_endbox(start_position, goal_position, uav.max_velocity, uav.drone_id,dynamic_time_span=True)
        uav.start_hitbox = gen_startbox(start_position, goal_position, uav.max_velocity, uav.drone_id,dynamic_time_span=False)
        uav.hit_boxes = [[uav.end_hitbox, uav.start_hitbox]]
        theta_star.hit_boxes.extend([[uav.end_hitbox, uav.start_hitbox]])

    # Path planning and smoothing
    logging.info("Path planning ...")

    start_time_point = time.time()  # Start time measurement
    for i, uav in enumerate(tqdm(uavs, desc="Planning paths")):
        start, goal = end_nodes[i]
        start_position = theta_star.graph.nodes[start]["coords"]
        uav.plan_path(uavs, planner=theta_star, start=start, goal=goal)

        if uav.path is None:
            print('no path is found')
            uav.start_hitbox = gen_startbox(start_position, drone_id=uav.drone_id)
            uav.hit_boxes = [[uav.start_hitbox]]
            TEST_STATS['solved'] = False


    TEST_STATS['LoS'], LoS_UAVs = get_LoS(uavs)
    if TEST_STATS['LoS'] == True:
        TEST_STATS['solved'] = False

    TEST_STATS['ftime'] = round(sum([uav.total_flight_time for uav in uavs]),2)
    TEST_STATS['simple_ftime'] = round(sum([uav.total_simple_flight_time for uav in uavs]),2)
    
    end_time_point = time.time()  # End time measurement

    # Calculate the elapsed time
    elapsed_time = end_time_point - start_time_point

    TEST_STATS['comp_time'] = round(elapsed_time,2)

    # Print the time taken
    print(f"scen {scen + 1}: Planning took {elapsed_time:.4f} seconds")
    logging.info("Path planning finished !")
    store_results(TEST_STATS, configs.stats_storage_path)

    # Visualization
    if visualize:
        logging.info("Visualization ...")
        animate_raylib(uavs, buildings, graph=theta_star.graph, end_nodes=end_nodes, bounds=configs.bounds, show_only=LoS_UAVs)
