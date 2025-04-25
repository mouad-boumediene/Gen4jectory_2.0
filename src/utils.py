import numpy as np
import pandas as pd
import networkx as nx
import numpy as np
from src.Box import ObstacleBox
from src.Box import Endbox, Box, Building
import pyray as pr
import logging
from matplotlib import pyplot as plt
from src import configs
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Node:
    def __init__(self, point, parent, time=0):
        self.point = point
        self.parent = parent
        self.time = time


class MapManager:
    def __init__(self, bounds, n_obstacles=10):
        self.bounds = bounds  # Bounds should be a list of tuples for each axis: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        self.obstacles = []
        self.generate_initial_obstacles(n_obstacles)

    def generate_initial_obstacles(self, n_obstacles):
        for _ in range(n_obstacles):
            # Randomly generate obstacle center within bounds
            center = [np.random.uniform(low, high) for low, high in self.bounds]
            # Randomly determine the size of the obstacle for each dimension
            size = [np.random.uniform(1, 5) for _ in range(3)]  # Generate size for x, y, and z dimensions
            self.obstacles.append(ObstacleBox(center, *size))


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def are_points_close_to_any_building(
    points: np.ndarray, 
    buildings: list, 
    threshold: float = 1.2
) -> bool:
    """
    Check if any point is within a threshold distance of any building.

    Args:
        points (np.ndarray): Array of points with shape (N, 3) where N is the number of points.
        buildings (list): List of Building objects.
        threshold (float): Distance threshold in meters (default is 1.2).

    Returns:
        bool: True if any point is within the threshold distance of any building, False otherwise.
    """
    points = np.array(points)
    
    for building in buildings:
        box_min, box_max = building.min_corner, building.max_corner
        
        # Expand the bounding box by the threshold distance
        expanded_min = box_min - threshold
        expanded_max = box_max + threshold
        
        # Check if any point lies within the expanded bounding box
        inside = np.all((points >= expanded_min) & (points <= expanded_max), axis=1)
        
        if np.any(inside):
            return True
    
    return False

def create_grid_3D(bounds: list[tuple[float, float]], resolution: float) -> np.ndarray:
    """
    Generate a 3D grid of points within the specified bounds at a given resolution.

    Args:
        bounds (list[tuple[float, float]]): List of tuples representing the min and max bounds for x, y, z axes.
        resolution (float): Distance between adjacent points in the grid.

    Returns:
        np.ndarray: A 3D grid of points.
    """
    x = np.arange(bounds[0][0], bounds[0][1] + 1, resolution)
    y = np.arange(bounds[1][0], bounds[1][1] + 1, resolution)
    z = np.arange(bounds[2][0], bounds[2][1] + 1, configs.z_resolution)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    return grid

def is_point_inside_obstacle(point: tuple[float, float, float], center: tuple[float, float, float], size: tuple[float, float, float]) -> bool:
    """
    Determine if a point is inside a rectangular obstacle.

    Args:
        point (tuple[float, float, float]): The coordinates of the point to check.
        center (tuple[float, float, float]): The center of the obstacle.
        size (tuple[float, float, float]): The size dimensions of the obstacle.

    Returns:
        bool: True if the point is inside the obstacle, False otherwise.
    """
    half_size = np.array(size) / 2.0
    point = np.array(point)
    center = np.array(center)
    return np.all(np.abs(point - center) < half_size)




def check_hitbox_intersection(hit_boxes,uavs, drone_id, new_hitbox, static_obstacles = None):
    if are_points_close_to_any_building(new_hitbox.collisionSpheres,static_obstacles,configs.collision_radius):
        return True
    # Retrieve new hitbox time details from the object
    start_time_new, duration_new = new_hitbox.start_time, new_hitbox.duration
    end_time_new = start_time_new + duration_new
    #print('checking ', len(list(np.array(hit_boxes).flatten())))
    for j, uav in enumerate(uavs):
        if uav.hit_boxes is None:
            continue
        if drone_id == uav.drone_id:
            continue
        else:

            for _, list_hitboxes in enumerate(uav.hit_boxes):
                for _, hitbox in enumerate(list_hitboxes):
                    if hitbox is None:
                        continue
                    if isinstance(new_hitbox, Endbox) and isinstance(hitbox, Endbox):
                        continue

                    #distance = np.linalg.norm(hitbox.center - new_hitbox.center)
                    #if distance > (hitbox.radius*1.5 + new_hitbox.radius*1.5):
                    #    # Skip to the next iteration if the spheres overlap
                    #    continue

                    # Retrieve old hitbox time details from the object
                    start_time_old, duration_old = hitbox.start_time, hitbox.duration
                    end_time_old = start_time_old + duration_old
                    # Check for spatial intersection and time overlap
                    if end_time_new >= start_time_old and start_time_new <= end_time_old:
                        if is_collide(new_hitbox, hitbox):
                            return True,[j] # false means thes two 4D hitboxes intersection

    return False







def closest_points_fromlists(p1, p2):
    """Finds the closest pair of points between two lists of points."""
    points1 = np.array(p1)
    points2 = np.array(p2)

    # Compute all pairwise distances using broadcasting
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]


    distances = np.linalg.norm(diff, axis=2)
    
    # Find the indices of the minimum distance
    
    min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    closest_pair = (points1[min_index[0]], points2[min_index[1]])
    
    # Retrieve the closest pair and the minimum distance
    min_distance = distances[min_index]

    return min_distance,closest_pair


def is_collide(box1:Box, box2:Box):
    
    dist, _= closest_points_fromlists(box1.collisionSpheres, box2.collisionSpheres)
    if dist <= configs.collision_radius*2:
        return True
    return False

def generate_points_along_line(P1, P2, point_spacing=configs.collision_spheres_dist):
    P1 = np.array(P1)
    P2 = np.array(P2)
    D = P2 - P1
    
    # Calculate the distance between P1 and P2
    distance = np.linalg.norm(D)
    
    # Determine the number of points based on the distance and point_spacing
    if distance == 0:
        # P1 and P2 are the same point
        return np.array([P1])
    
    n = int(distance / point_spacing) + 1
    
    # Handle the case where n == 1
    if n == 1:
        return np.array([P1, P2])
    
    # Generate points along the line, including P1 and P2
    points = [P1 + (i / (n - 1)) * D for i in range(n)]
    
    # Ensure P2 is included as the last point
    points[-1] = P2  # Ensure the last point is exactly P2
    
    return np.array(points)

def gen_endbox(start_position, goal_position, velocity, drone_id,start_time=0, dynamic_time_span = False):

    if dynamic_time_span:
        return Endbox( drone_id=drone_id, 
                        center=goal_position, 
                        start_time=(distance(goal_position,start_position)/velocity)/2 + start_time ,
                        duration=np.inf,
                        edge_duration=np.inf
                        )
    else:
        return Endbox( drone_id=drone_id, 
                        center=goal_position, 
                        start_time=start_time,
                        duration=np.inf,
                        edge_duration=np.inf
                        )



def gen_startbox(start_position, goal_position=None, velocity=0, drone_id=None,start_time=0, dynamic_time_span= False):
    if dynamic_time_span:

        return Endbox( drone_id=drone_id, 
                        center=start_position, 
                        start_time=start_time,
                        duration=(distance(goal_position,start_position)/velocity)/2+start_time,
                        edge_duration=(distance(goal_position,start_position)/velocity)/2+start_time )
        
    else:
        

        return Endbox( drone_id=drone_id, 
                    center=start_position, 
                    start_time=start_time,
                    duration=np.inf,
                    edge_duration=np.inf)#(distance(goal_position,start_position)/velocity)/2 )


def generate_buildings(bounds=configs.bounds, num_buildings=2, base_size=5):
    buildings = []
    for _ in range(num_buildings):
        # Calculate safe position limits to ensure the building stays within bounds
        min_x = bounds[0][0] + base_size / 2
        max_x = bounds[0][1] - base_size / 2
        min_y = bounds[1][0] + base_size / 2
        max_y = bounds[1][1] - base_size / 2

        # Random position within the safe limits
        position_x = np.random.uniform(min_x, max_x)
        position_y = np.random.uniform(min_y, max_y)

        # Probability-based height: most buildings are low, few are tall
        height = np.random.uniform(configs.bounds[2][1]/2, configs.bounds[2][1])  # Tall buildings
        
        # Position the building with the base centered and height along the Z-axis
        position = [position_x, position_y, height / 2]
        
        # Add the building's position and size to the list
        buildings.append(Building(position,base_size,base_size,height))
    
    return buildings

def plot_box(ax, box, color='blue', alpha=0.25):
    vertices = np.array(box.vertices)
    faces = [[vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]], 
             [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]], 
             [vertices[j] for j in [0, 3, 7, 4]], [vertices[j] for j in [1, 2, 6, 5]]]
    ax.add_collection3d(Poly3DCollection(faces, linewidths=1, edgecolors='r', alpha=alpha, color = color ))


def get_LoS(uavs):
    for i, uav in enumerate(uavs):
        for _, list_hitboxes in enumerate(uav.hit_boxes):
            for _, hitbox in enumerate(list_hitboxes):
                if uav.hit_boxes is None:
                    continue

                # Retrieve old hitbox time details from the object
                start_time_old, duration_old = hitbox.start_time, hitbox.duration
                end_time_old = start_time_old + duration_old
                for j in range(i + 1, len(uavs)):
                    uav2 = uavs[j]
                    if uav2.hit_boxes is None:
                        continue
                    if uav.drone_id == uav2.drone_id:
                        continue
                    else:

                        for _, list_hitboxes2 in enumerate(uav2.hit_boxes):
                            for _, new_hitbox in enumerate(list_hitboxes2):
                                if new_hitbox is None:
                                    continue
                                if isinstance(new_hitbox, Endbox) and isinstance(hitbox, Endbox):
                                    continue

                                #distance = np.linalg.norm(hitbox.center - new_hitbox.center)
                                #if distance > (hitbox.radius*1.1 + new_hitbox.radius*1.1):
                                #    # Skip to the next iteration if the spheres overlap
                                #    continue

                                # Retrieve old hitbox time details from the object
                                start_time_new, duration_new = new_hitbox.start_time, new_hitbox.duration
                                end_time_new = start_time_new + duration_new
                                # Check for spatial intersection and time overlap
                                if end_time_new >= start_time_old and start_time_new <= end_time_old:
                                    if is_collide(new_hitbox, hitbox):
                                        print('collision between\n')
                                        print(new_hitbox.center, hitbox.center)
                                        print(j,i)
                                        return True,[i,j] # false means thes two 4D hitboxes intersection
    return False, None


def draw_box_between_points(vertices):
    # Draw small spheres at each vertex
    for vertex_id,vertex_coords in enumerate(vertices):
        pr.draw_sphere(pr.Vector3(vertex_coords[0], vertex_coords[1], vertex_coords[2]), 0.05, pr.MAROON)
    vertices = [pr.Vector3(x, y, z) for x, y, z in vertices]
    
    # Draw lines for the left face
    pr.draw_line_3d(vertices[0], vertices[1], pr.RED)
    pr.draw_line_3d(vertices[1], vertices[2], pr.RED)
    pr.draw_line_3d(vertices[2], vertices[3], pr.RED)
    pr.draw_line_3d(vertices[3], vertices[0], pr.RED)

    

    # Draw lines for the right face
    pr.draw_line_3d(vertices[4], vertices[5], pr.RED)
    pr.draw_line_3d(vertices[5], vertices[6], pr.RED)
    pr.draw_line_3d(vertices[6], vertices[7], pr.RED)
    pr.draw_line_3d(vertices[7], vertices[4], pr.RED)

    # draw front face
    # Draw lines connecting left and right faces
    pr.draw_line_3d(vertices[0], vertices[4], pr.RED)
    pr.draw_line_3d(vertices[1], vertices[5], pr.RED)
    pr.draw_line_3d(vertices[2], vertices[6], pr.RED)
    pr.draw_line_3d(vertices[3], vertices[7], pr.RED)




# Define a custom filter class
class CustomFilter(logging.Filter):
    def __init__(self, levels_to_disable):
        self.levels_to_disable = levels_to_disable

    def filter(self, record):
        return record.levelno not in self.levels_to_disable

# Function to configure logging and set filters
def configure_logging(disabled_levels):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get the root logger
    root_logger = logging.getLogger()
    
    # Create and add the custom filter
    custom_filter = CustomFilter(disabled_levels)
    root_logger.addFilter(custom_filter)

def order_file_by_scene(file_path: str) -> None:
    # Read the content from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the header and the data lines
    header = lines[0].strip()
    data_lines = [line.strip() for line in lines[1:] if line.strip()]
    
    # Sort the data lines based on the scene (first column)
    sorted_data_lines = sorted(data_lines, key=lambda x: int(x.split(',')[0]))
    
    # Extract the scene numbers from the sorted data lines
    scene_numbers = [int(line.split(',')[0]) for line in sorted_data_lines]
    
    # Find the smallest and largest scene numbers
    smallest_scene = scene_numbers[0]
    largest_scene = scene_numbers[-1]
    
    # Check for missing scene numbers
    missing_scenes = [i for i in range(smallest_scene, largest_scene + 1) if i not in scene_numbers]
    
    # Print or handle the missing scenes as needed
    print(f"Missing scene numbers: {missing_scenes}")
    
    # Combine the header and the sorted data lines
    sorted_content = '\n'.join([header] + sorted_data_lines)
    
    # Write the sorted content back to the file
    with open(file_path, 'w') as file:
        file.write(sorted_content + '\n')


def compute_segment_intersections(nodes, end_nodes):
    """
    nodes: dict of node_id -> {"coords": (x, y)}
    end_nodes: list of (start_id, goal_id) tuples

    Returns: list of lists, one per UAV i:
             - If UAV i has intersections with others, inner list is
               [(j, px, py), ...] for each UAV j ≠ i whose segment crosses i
             - If no intersections, inner list is [None]
    """
    def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0:
            return None
        det1 = x1*y2 - y1*x2
        det2 = x3*y4 - y3*x4
        px = (det1*(x3 - x4) - (x1 - x2)*det2) / denom
        py = (det1*(y3 - y4) - (y1 - y2)*det2) / denom
        return px, py

    def on_segment(pt, a, b):
        px, py = pt
        ax, ay = a
        bx, by = b
        return (min(ax, bx) <= px <= max(ax, bx)) and (min(ay, by) <= py <= max(ay, by))

    # build list of (start_coords, goal_coords) for each UAV
    paths = [(nodes[s]["coords"][:2], nodes[g]["coords"][:2]) for s, g in end_nodes]
    n = len(paths)
    all_inters = []

    for i in range(n):
        a, b = paths[i]
        x1, y1 = a; x2, y2 = b
        inner = []
        for j in range(n):
            if j == i:
                continue
            c, d = paths[j]
            x3, y3 = c; x4, y4 = d

            pt = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
            if pt is not None and on_segment(pt, a, b) and on_segment(pt, c, d):
                px, py = pt
                inner.append((j, px, py))

        # if no intersections, record a single None
        if not inner:
            inner = [None]

        all_inters.append(inner)

    return all_inters


