"""
Module: planners.py
Description:
    Theta* path-planning for UAVs in 3D:
      - Builds a 3D grid graph excluding obstacle nodes.
      - Performs Theta* search with 4D hitbox collision checking.
      - Reconstructs timed UAV trajectories and hit-box sequences.
"""
import numpy as np
from src.utils import create_grid_3D
import numpy as np
import networkx as nx

from src.utils import is_point_inside_obstacle
from heapq import heappush, heappop
from src.utils import check_hitbox_intersection
from src.Box import Hitbox, Endbox
from src.agent import UAV
from src import configs
import logging
import copy
import time

from matplotlib import pyplot as plt


class ThetaStar:
    def __init__(self,obstacles, bounds, resolution = configs.resolution):
        self.obstacles = obstacles
        self.bounds = bounds
        
        self.hit_boxes:list[Hitbox] = []
        self.end_hit_boxes:list[Hitbox] = []
        logging.info("creating a graph")
        grid=create_grid_3D(bounds,resolution)
        filtered_grid = self._remove_nodes_in_obstacles(grid, obstacles)

        self.graph:nx.Graph = self._create_graph(filtered_grid, resolution)
        logging.info("created a graph")

        
    def _remove_nodes_in_obstacles(self, grid, obstacles):
        filtered_grid = []
        for point in grid:
            inside_any_obstacle = any(is_point_inside_obstacle(point = point, 
                                                               center = obs.center, 
                                                               size = (obs.length, obs.width, obs.height)) for obs in obstacles)
            
            
            if not inside_any_obstacle:
                filtered_grid.append(tuple(point))
            else:
                logging.debug('removed node in obstacles ')
        return filtered_grid

    
    def _create_graph(self, grid, resolution = configs.resolution)->nx.Graph:
        G = nx.Graph()
        # Create a mapping from points to unique integer identifiers
        point_to_id = {point: i for i, point in enumerate(grid)}

        for point, node_id in point_to_id.items():
            G.add_node(node_id, coords=point, velocity=0.0, time = 0.0)

        for node in G.nodes:
            x, y, z = G.nodes[node]['coords']
            neighbors = [
                            # Direct neighbors (6 directions)
                            (x + resolution, y, z), (x - resolution, y, z),
                            (x, y + resolution, z), (x, y - resolution, z),
                            (x, y, z + configs.z_resolution), (x, y, z - configs.z_resolution)]
            """,
                            
            
             
                            # Diagonal neighbors in XY plane (4 directions)
                            (x + resolution, y + resolution, z), (x - resolution, y - resolution, z),
                            (x + resolution, y - resolution, z), (x - resolution, y + resolution, z),
                            
                            # Diagonal neighbors in XZ plane (4 directions)
                            (x + resolution, y, z + resolution), (x - resolution, y, z - resolution),
                            (x + resolution, y, z - resolution), (x - resolution, y, z + resolution),
                            
                            # Diagonal neighbors in YZ plane (4 directions)
                            (x, y + resolution, z + resolution), (x, y - resolution, z - resolution),
                            (x, y + resolution, z - resolution), (x, y - resolution, z + resolution),
                            
                            # Diagonal neighbors in XYZ space (8 directions)
                            (x + resolution, y + resolution, z + resolution), (x - resolution, y - resolution, z - resolution),
                            (x + resolution, y - resolution, z - resolution), (x - resolution, y + resolution, z + resolution),
                            (x + resolution, y + resolution, z - resolution), (x - resolution, y - resolution, z + resolution),
                            (x + resolution, y - resolution, z + resolution), (x - resolution, y + resolution, z - resolution)
                        ]"""
            for neighbor in neighbors:
                if neighbor in point_to_id:  # Check if the neighbor exists in the grid
                    neighbor_id = point_to_id[neighbor]  # Map neighbor coordinate to node identifier
                    G.add_edge(node, neighbor_id)
        
        return G

    
    def visualize_graph_3d(self,G=None):
        if G is None:
            G = self.graph
        pos = {node: G.nodes[node]['coords'] for node in G.nodes}
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for edge in G.edges:
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, color='b')
        
        x_nodes = [pos[node][0] for node in G.nodes]
        y_nodes = [pos[node][1] for node in G.nodes]
        z_nodes = [pos[node][2] for node in G.nodes]
        ax.scatter(x_nodes, y_nodes, z_nodes, color='r', s=20)
        
        plt.show()

    def reconstruct_path(self, goal, parents, uav:UAV, path_graph):
        current = goal
        path = [current]
        hit_boxes = copy.deepcopy(uav.hit_boxes)

        while current in parents:
            current = parents[current]
            path.append(current)
            
        path.reverse()

        total_duration = 0
        total_simple_duration = 0

        if len(path)>2:
            for i in range(len(path)-1):
                
                segment_hit_boxes, hit_box = path_graph.get_edge_data(path[i],path[i+1])['box']
                #splitted_segment_hitboxes = hit_box.split_box()
                #segment_hit_boxes = self.handle_seperation(splitted_segment_hitboxes)
                hit_boxes.append(segment_hit_boxes)
                total_duration += hit_box.duration
                total_simple_duration += hit_box.simple_duration

        else:
            segment_hit_boxes,hit_box = path_graph.get_edge_data(path[0],path[1])['box']
            #splitted_segment_hitboxes = hit_box.split_box()
            #segment_hit_boxes = self.handle_seperation(splitted_segment_hitboxes)
            hit_boxes.append(segment_hit_boxes)
            total_duration += hit_box.duration
            total_simple_duration += hit_box.simple_duration
        
        #hit_boxes.append([uav.end_hitbox, uav.start_hitbox])
        uav.total_flight_time = total_duration
        uav.total_simple_flight_time = total_simple_duration
        return path, hit_boxes


    def handle_seperation(self, segment_hit_boxes:list):
        hit_boxes = copy.deepcopy(segment_hit_boxes)
        new_hit_boxes = []
        for i, hit_box in enumerate(segment_hit_boxes):
            if len(hit_boxes) <= 1:
                new_hit_boxes.append(hit_box)
                break
            if i == 0 :
                hit_box.duration = hit_boxes[i].duration + hit_boxes[i+1].duration
                new_hit_boxes.append(hit_box)
            elif i == len(segment_hit_boxes)-1:
                hit_box.duration = hit_boxes[i].duration + hit_boxes[i-1].duration
                hit_box.start_time = hit_box.start_time - hit_boxes[i-1].duration
                new_hit_boxes.append(hit_box)
                return new_hit_boxes
            else:
                hit_box.duration = hit_boxes[i-1].duration + hit_boxes[i].duration + hit_boxes[i+1].duration
                hit_box.start_time = hit_box.start_time - hit_boxes[i-1].duration
                new_hit_boxes.append(hit_box)
        return new_hit_boxes
        
    
    def plan(self, uav:UAV, start, goal, start_time = 0, uavs=[], intersections = [None]):
        path_graph = nx.Graph()
        path_graph.add_node(start, coords = self.graph.nodes[start]['coords'])
        
        
        uav.start_position = np.array(self.graph.nodes[start]['coords'])
        uav.goal_position = np.array(self.graph.nodes[goal]['coords'])
        # calculate the euclidian distance between two nodes
        def heuristic(graph, node1, node2):
            coords1, coords2 = graph.nodes[node1]["coords"], graph.nodes[node2]["coords"]
            return np.linalg.norm(np.array(coords1) - np.array(coords2))
        
        open_set = []
        # at first push the start node
        heappush(open_set, (0, 0, start, start))
        came_from = {}
        
        # reset the time and final velocity for all nodes in the graph
        nx.set_node_attributes(self.graph, np.inf, 'time')
        nx.set_node_attributes(self.graph, 0, 'v_f')

        self.graph.nodes[start]['time'] = start_time


        # set the g_score 
        g_score = {node: float('inf') for node in self.graph.nodes}
        g_score[start] = 0
        iteration = 0

        # search for the least cost path
        while open_set:
                        

            # pop the node with least cost
            _, _, current, parent= heappop(open_set)            

            # if you poped the goal node (congratulations)
            if current == goal:
                
                #self.visualize_graph_3d(path_graph)
                path, hit_boxes = self.reconstruct_path(goal, came_from, uav, path_graph)




                
                self.hit_boxes.extend(hit_boxes)
                return path, hit_boxes

            # loop over all neighbors of the current node you poped from the heap
            for neighbor in self.graph.neighbors(current):
                #print(len(list(self.graph.neighbors(current))))
                #print(f"Node {current} has {len(list(self.graph.neighbors(current)))} neighbors")
                

                # no going back to parent
                if parent == neighbor:
                    continue
                
                if parent == current:
                    # if the current node is it's own parent 
                    # if this parent is the start node (the parent of the start node is the start node itself)
                    # use the uav max velocity to get the flight time
                    flight_time = heuristic(self.graph, current, neighbor)/uav.max_velocity
                    simple_flight_time = flight_time
                    v_f = uav.max_velocity

                    flight_time_ls = heuristic(self.graph, parent, neighbor)/uav.max_velocity
                    simple_flight_time_ls = flight_time_ls
                    v_f_ls = uav.max_velocity
                else: 
                    # the flight time is calculated using the flight model
                    # given the current node(i.e parent), it's parent (grandparent) and the neighbor(i.e. child)
                    
                    v_0, v_f, flight_time, simple_flight_time = uav.calculate_flight_time(self.graph.nodes[current]['v_f'], 
                                                                    self.graph.nodes[parent]['coords'], 
                                                                    self.graph.nodes[current]['coords'], 
                                                                    self.graph.nodes[neighbor]['coords'])
                    
                    
                    # the grand_parent node is needed if the line_of_sight(ls) is true
                    if parent == start: # it the parent is the start node its parent will be itself
                        flight_time_ls = heuristic(self.graph, parent, neighbor)/uav.max_velocity
                        simple_flight_time_ls = flight_time_ls
                        v_f_ls = uav.max_velocity
                    else: # getting the parent of hte current parent and calculate the flight time and v_f for ls
                        parent_parent = came_from[parent]
                        v_0_ls, v_f_ls, flight_time_ls, simple_flight_time_ls = uav.calculate_flight_time(self.graph.nodes[parent]['v_f'], 
                                                                        self.graph.nodes[parent_parent]['coords'], 
                                                                        self.graph.nodes[parent]['coords'], 
                                                                        self.graph.nodes[neighbor]['coords'])
                
                # check if this neighbor have direct line of sight
                # to the parent of the current node
                is_line_of_sight, split_hit_boxes = self.line_of_sight(uav,uavs, parent, neighbor, flight_time_ls, simple_flight_time_ls,intersections)
                if is_line_of_sight:
                    tentative_g_score = g_score[parent] + flight_time_ls

                    # if going throught the neighbor node is 
                    # better than going throught the previously expanded nodes
                    if tentative_g_score < g_score[neighbor]:
                        # calculate the starting time of this 4D box
                        nearest_node_time = self.graph.nodes[parent]['time']
                        
                        
                        # check intersection of new 4D box and other boxes from other drones 
                        #if not check_hitbox_intersection(self.hit_boxes, uav.drone_id, hit_box, self.obstacles):
                        # get this neighbor node info to push in the heap
                        came_from[neighbor] = parent
                        self.graph.nodes[neighbor]['v_f'] = v_f_ls # store final vel of the uav
                        self.graph.nodes[neighbor]['time'] = nearest_node_time + flight_time_ls
                        

                        # the temporary g_score is now the g_score for the neighbor node
                        g_score[neighbor] = tentative_g_score 

                        # the f_score is the g_score + direct flight time to the goal
                        # from the neighbor node
                        f_score = tentative_g_score + heuristic(self.graph, neighbor, goal)/uav.max_velocity

                        # push the neighbor node info into the heap
                        heappush(open_set, (f_score, tentative_g_score, neighbor, parent))
                        # store the hitbox to be used in reconstruction
                        path_graph.add_node(neighbor, coords = self.graph.nodes[neighbor]['coords']) 
                        path_graph.add_edge(parent, neighbor, box = split_hit_boxes)
                        
                        
                else:
                    
                    tentative_g_score = g_score[current] + flight_time

                    # if going throught the neighbor node is 
                    # better than going throught the previously expanded nodes
                    if tentative_g_score < g_score[neighbor]:

                        # calculate the starting time of this 4D box
                        nearest_node_time = self.graph.nodes[current]['time']

                        # create a 4D box
                        hit_box = Hitbox(self.graph,
                                        uav.drone_id,
                                        self.graph.nodes[current]["coords"],
                                        self.graph.nodes[neighbor]["coords"],
                                        start_time=nearest_node_time,
                                        duration=flight_time,
                                        start_node=current,
                                        end_node=neighbor,
                                        simple_duration=simple_flight_time)
                        
                        # check if the box intersects with other boxes
                        # (boxes for the same uav can intersect)
                        # mesure how much time splitting boxes and handling seperation take
                        split_box_time = time.time()
                        splitted_segment_hitboxes = hit_box.split_box_new(intersections,configs.intersection_collision_radius)
                        segment_hit_boxes = self.handle_seperation(splitted_segment_hitboxes)
                        #print(f"segment hit boxes {len(segment_hit_boxes)}")

                        is_intersect = False
                        for new_hit_box in segment_hit_boxes:
                            if check_hitbox_intersection(self.hit_boxes,uavs, uav.drone_id, new_hit_box, self.obstacles):
                                is_intersect = True
                                break
                        if not is_intersect:
                            came_from[neighbor] = current
                            self.graph.nodes[neighbor]['v_f'] = v_f
                            self.graph.nodes[neighbor]['time'] = nearest_node_time + flight_time

                            # the temporary g_score is now the g_score for the neighbor node
                            g_score[neighbor] = tentative_g_score

                            # push the neighbor node info into the heap
                            f_score = tentative_g_score + heuristic(self.graph, neighbor, goal)/uav.max_velocity
                            heappush(open_set, (f_score, tentative_g_score, neighbor, current))
                            path_graph.add_node(neighbor, coords = self.graph.nodes[neighbor]['coords'])
                            path_graph.add_edge(current, neighbor, box = [segment_hit_boxes, hit_box])

            iteration += 1
        

        
        return None, None
    
    
    def line_of_sight(self,uav:UAV,uavs:list,node1, node2, flight_time, simple_flight_time,intersections = [None]):
        nearest_node_time = self.graph.nodes[node1]['time']
        if node1 == node2:
            raise
        hit_box = Hitbox(self.graph, 
                         uav.drone_id, 
                         self.graph.nodes[node1]["coords"], 
                         self.graph.nodes[node2]["coords"], 
                         start_time=nearest_node_time, 
                         duration=flight_time, 
                         simple_duration=simple_flight_time,
                         start_node= node1, 
                         end_node=node2)
                         
        splitted_segment_hitboxes = hit_box.split_box_new(intersections,configs.intersection_collision_radius)
        segment_hit_boxes = self.handle_seperation(splitted_segment_hitboxes)
        for new_hit_box in segment_hit_boxes:
            if check_hitbox_intersection(self.hit_boxes,uavs, uav.drone_id, new_hit_box, self.obstacles):
                return False,None
        return True,[segment_hit_boxes, hit_box]

