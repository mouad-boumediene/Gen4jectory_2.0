"""
Module: agent.py
Description:
    Defines the UAV and UAVGenerator classes for:
      - Simulating UAV physical flight parameters via UAVFlightModel.
      - Generating random or fixed start/goal endpoints on a NetworkX graph.
      - Creating hit-boxes and computing distances while avoiding obstacles.
"""
import random
import uuid
import networkx as nx
from src.utils import distance, are_points_close_to_any_building, gen_endbox, gen_startbox
import pyray as pr
import numpy as np
from src.drone_model import UAVFlightModel
from src import configs
import logging
class UAV:
    def __init__(self,drone_id, max_thrust, mass, csa, c_d,theta_limit):
        self.m = mass # Mass of the UAV, kg
        self.max_thrust = max_thrust
        self.csa = csa
        self.c_d = c_d
        self.theta_limit = theta_limit
        self.max_velocity = None # m/s
        self.initial_velocity = 0.0
        self.end_hitbox = []
        self.start_hitbox = []
        self.start_position = None
        self.goal_position = None
        self.start = None
        self.goal = None
        
        
        self.path = []
        self.current_time = 0  # Starting at time 0, adjust as necessary
        self.node_reservations = []
        self.hit_boxes = []
        self.trajectory_segment_mapping = None
        self.trajectory = []
        self.drone_id = drone_id

        

        self.current_position = None
        self.current_waypoint_index = 0
        self.t = 0  # Interpolation factor, ranges from 0 to 1

        
        self.color = pr.DARKGRAY
        
        self.total_simple_flight_time = np.inf
        self.total_flight_time = np.inf
        self.start_time = 0


        self.flight_parameters =  UAVFlightModel(   m=self.m, f_t=self.max_thrust, 
                                                    csa=self.csa, c_d=self.c_d, 
                                                    g=9.81,rho=1.225, theta_limit=self.theta_limit, 
                                                    v_0=0.0, thrust_min_percent=2, delta_time=0.1
                                                )
        
        self.max_velocity = self.flight_parameters.v_max
        
    
    def reset_position(self):
        self.current_position = self.start_position
        self.current_waypoint_index = 0
        self.t = 0
        
    def update_position(self, position):
        self.current_position = position

    def plan_path(self,uavs, planner , start, goal, endpoints_intersections)->None:
        start_time = 0
        for start_time in np.arange(0, 5*61, 1):
            if start_time != 0:
                print(f'replanning for agent {self.drone_id}, start time : {start_time} ')
                self.start_time = start_time
            self.path, self.hit_boxes = planner.plan(self, start, goal, start_time = start_time, uavs=uavs, intersections=endpoints_intersections)

            if self.path is not None:
                return self.path
            start_position = planner.graph.nodes[start]["coords"]
            goal_position = planner.graph.nodes[goal]["coords"]
            self.end_hitbox = gen_endbox(start_position,goal_position, self.max_velocity, self.drone_id, start_time)
            self.start_hitbox = gen_startbox(start_position,goal_position, self.max_velocity, self.drone_id,start_time)
            self.hit_boxes = [[self.end_hitbox, self.start_hitbox]]
        return None
                        
            
    
    def calculate_flight_time(self, v_f_parent,  grand_parent, parent, child):
        # Initialize waypoints and initial velocities
        waypoints = [grand_parent, parent, child]

        # Simulate flight
        v_0, v_f, flight_time, simple_flight_time = self.flight_parameters.simulate_flight(v_f_parent,waypoints)
        return v_0, v_f, flight_time, simple_flight_time
    


        

        
            


    

    



    




class UAVGenerator:
    def __init__(self):
        pass
    
    def gen_random_endpoints(
        self, 
        G: nx.Graph, 
        bounds: list[tuple[float, float]], 
        num_agents: int, 
        obstacles: list = None, 
        min_distance: float = configs.collision_radius*4
    ) -> list[tuple[int, int]]:
        """
        Generate random start and goal positions for UAVs.

        Args:
            G (nx.Graph): Graph to add UAV nodes and edges.
            bounds (list[tuple[float, float]]): Bounding box for random positions.
            num_agents (int): Number of UAVs to simulate.
            obstacles (list, optional): List of obstacles to avoid. Defaults to None.
            min_distance (float): Minimum distance between start, goal, and other nodes.

        Returns:
            list[tuple[int, int]]: List of start and goal node ID pairs.
        """
        used_locations = set()
        nodes = list(G.nodes(data=True))
        coords_to_id = {node[1]['coords']: node[0] for node in nodes}
        coordinates = []

        for _ in range(num_agents):
            while True:
                start = tuple(round(random.uniform(b[0], b[1]), 1) for b in bounds)
                goal = tuple(round(random.uniform(b[0], b[1]), 1) for b in bounds)

                if all(distance(start, loc) >= min_distance and distance(goal, loc) >= min_distance for loc in used_locations):
                    if distance(start, goal) >= min_distance:
                        if obstacles is None or (not are_points_close_to_any_building([start, goal], buildings=obstacles, threshold=configs.collision_radius*4)):
                            break

            start_id = len(G.nodes.keys())
            goal_id = len(G.nodes.keys()) + 1

            G.add_node(start_id, coords=start, velocity=0.0, time=0.0, node_type='S')
            G.add_node(goal_id, coords=goal, velocity=0.0, time=0.0, node_type='G')

            start_distances = [(node, distance(start, data['coords'])) for node, data in G.nodes(data=True)]
            goal_distances = [(node, distance(goal, data['coords'])) for node, data in G.nodes(data=True)]

            start_distances.sort(key=lambda x: x[1])
            goal_distances.sort(key=lambda x: x[1])

            G.add_edge(start_id, goal_id, weight=distance(start, goal), reservations=[])

            num_neighbors = 20
            for node, dist in start_distances[:num_neighbors]:
                G.add_edge(start_id, node, weight=dist, reservations=[])

            for node, dist in goal_distances[:num_neighbors]:
                G.add_edge(goal_id, node, weight=dist, reservations=[])

            coordinates.append((start_id, goal_id))
            used_locations.update([start, goal])

        G.remove_edges_from(nx.selfloop_edges(G))

        return coordinates

    
    def gen_fixed_endpoints(self,z=5):
        coordinates = [[(8,0,z),(8,15,z)], [(0,8,z),(15,8,z)]]
        return coordinates
    
 
    def gen_uav(self,drone_id) -> UAV:
        """
        Generate a UAV with random physical properties.

        Returns:
            UAV: An instance of the UAV class with generated attributes.
        """
        g = 9.81
        mass: float = round(random.uniform(2, 500), 2)  # Mass of the UAV in kg
        max_thrust: float = round(random.uniform(mass * g * 1.2, mass * g * 2))

        csa: float = round(0.007014 * mass + 0.496493, 2)  # Cross-sectional area in m^2
        c_d: float = round(random.uniform(0.6, 1.2), 2)  # Drag coefficient
        theta_limit: float = random.uniform(20, 40)  # Maximum tilt angle in degrees

        return UAV(drone_id,max_thrust, mass, csa, c_d, theta_limit)

