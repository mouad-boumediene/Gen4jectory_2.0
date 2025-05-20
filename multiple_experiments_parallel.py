"""
Module: multiple_experiments_parallel.py
Description:
    Parallel execution of UAV path-planning scenarios:
      - Seeds repeatability via `scen` as RNG seed.
      - Generates randomized city maps.
      - Plans paths with the Theta* solver and checks loss-of-separation (LoS).
      - Records per-scenario stats (CSV output via `store_results`).
      - Optionally animates individual runs using Raylib.
"""
from src.planners import ThetaStar
from src.agent import UAVGenerator, UAV
from src.utils import gen_endbox, gen_startbox, get_LoS, generate_buildings, compute_segment_intersections, generate_city
import numpy as np
from src import configs
import logging
from src.visualization_raylib import animate_raylib
from tqdm import tqdm
import random
import time
import csv
from concurrent.futures import ProcessPoolExecutor

def store_results(dictionary, filename, print_line=False):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        if print_line:
            print("Header:", ', '.join(dictionary.keys()))
            print("Line:  ", ', '.join(map(str, dictionary.values())))
        writer.writerow(dictionary)

def run_scenario(scen, visualize = False):
    np.random.seed(scen)
    random.seed(scen)
    logging.info(f"scenario {scen + 1}/{configs.num_iterations}")

    TEST_STATS = {
        'sim_num': scen,
        'num_agents': configs.num_agents,
        'solver': 'Theta*',
        'LoS': False,
        'solved': True,
        'comp_time': 0.0,
        'ftime': np.inf,
        'map_size': (configs.bounds[0][1], configs.bounds[1][1], configs.bounds[2][1]),
        'resolution': configs.resolution
    }

    # Map Creation
    logging.info("Creating a map ...")
    bounds = configs.bounds
    buildings = []  # Initialize buildings as an empty list
    #buildings = generate_buildings(num_buildings=configs.num_buildings, base_size=configs.builduing_base_size)
    buildings = generate_city(bounds, obstacle_density=0.0175, base_size_pct=0.025, min_height=5, max_height=configs.bounds[2][1]-configs.bounds[2][0])

    # Generating the Drones
    logging.info("Generating the Drones ...")
    uav_generator = UAVGenerator()
    uavs = [uav_generator.gen_uav(drone_id=i) for i in range(configs.num_agents)]

    # Generating start and goal positions
    logging.info("Generating starts and goals ...")
    theta_star = ThetaStar(buildings, bounds, resolution=configs.resolution)
    end_nodes = uav_generator.gen_random_endpoints(theta_star.graph, bounds, configs.num_agents, obstacles=buildings)
    # Compute intersections on the finite segments
    intersections = compute_segment_intersections(theta_star.graph.nodes, end_nodes)
    # Generate end boxes
    for i, uav in enumerate(uavs):
        start, goal = end_nodes[i]
        start_position = theta_star.graph.nodes[start]["coords"]
        goal_position = theta_star.graph.nodes[goal]["coords"]
        uav.end_hitbox = gen_endbox(start_position, goal_position, uav.max_velocity, uav.drone_id,dynamic_time_span=False)
        uav.start_hitbox = gen_startbox(start_position, goal_position, uav.max_velocity, uav.drone_id,dynamic_time_span=False)
        uav.hit_boxes = [[uav.end_hitbox, uav.start_hitbox]]
        theta_star.hit_boxes.extend([[uav.end_hitbox, uav.start_hitbox]])

    # Path planning and smoothing
    logging.info("Path planning ...")
    start_time_point = time.time()

    for i, uav in enumerate(uavs):
        start, goal = end_nodes[i]
        start_position = theta_star.graph.nodes[start]["coords"]
        direct_path_intersections = intersections[i]
        uav.plan_path(uavs, planner=theta_star, start=start, goal=goal, endpoints_intersections=direct_path_intersections)
        
        if uav.path is None:
            print('no path is found')
            uav.start_hitbox = gen_startbox(start_position, drone_id=uav.drone_id)
            uav.hit_boxes = [[uav.start_hitbox]]
            TEST_STATS['solved'] = False

    TEST_STATS['LoS'], LoS_UAVs = get_LoS(uavs)
    if TEST_STATS['LoS']:
        TEST_STATS['solved'] = False

    TEST_STATS['ftime'] = round(sum([uav.total_flight_time for uav in uavs]), 2)
    #TEST_STATS['simple_ftime'] = round(sum([uav.total_simple_flight_time for uav in uavs]), 2)

    elapsed_time = time.time() - start_time_point
    TEST_STATS['comp_time'] = round(elapsed_time, 2)

    print(f"scen {scen + 1}: Planning took {elapsed_time:.4f} seconds")
    #logging.info("Path planning finished !")
    store_results(TEST_STATS, configs.stats_storage_path)

    if visualize:
        logging.info("Visualization ...")
        animate_raylib(uavs, buildings, graph=theta_star.graph, end_nodes=end_nodes, bounds=configs.bounds, show_only=LoS_UAVs)

    return TEST_STATS

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    visualize = False
    #run_scenario(3,visualize=False)

    with ProcessPoolExecutor(max_workers=5) as executor:
        #futures = executor.submit(run_scenario, 3, visualize)
        futures = [executor.submit(run_scenario, scen, visualize) for scen in range(configs.num_iterations)]
        for future in tqdm(futures, desc="Running scenarios"):
            future.result()  # Ensure each future is completed

