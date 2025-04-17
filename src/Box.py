import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import networkx as nx
from scipy.spatial.distance import euclidean
from src import configs
from math import ceil

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

class Box(ABC):
    """ Abstract base class to define the structure of 3D boxes. """
    def __init__(self, center, length, width, height):
        self.center = np.array(center)
        self.length = length
        self.width = width
        self.height = height
        self.vertices = []
        self.gen_box()
    
    @abstractmethod
    def gen_box(self):
        """ Method to generate the vertices of the box. """
        pass

@dataclass
class Endbox(Box):
    drone_id: str
    center: any
    start_time: any
    duration: any
    width: float = configs.buffer_zone_size
    height: float = configs.buffer_zone_size
    color:str = 'cyan'
    length:float = configs.buffer_zone_size
    edge_duration: float = None
    axes : any = None
    projection : any = None
    radius: any = None
    collisionSpheres: any = None

    def __post_init__(self):
        # Calculate the initial center position and then call the parent class's init
        super().__init__(self.center, self.length, self.width, self.height)
        self.gen_box()
        
    
    def calculate_radius(self):
        return np.max(np.linalg.norm(self.vertices - self.center, axis=1))

    def gen_box(self):
        # Calculate the half dimensions
        half_length = self.length / 2
        half_width = self.width / 2
        half_height = self.height / 2

        # Create 8 vertices of the box
        self.vertices = [
            self.center + np.array([half_length, half_width, half_height]),
            self.center + np.array([half_length, -half_width, half_height]),
            self.center + np.array([half_length, -half_width, -half_height]),
            self.center + np.array([half_length, half_width, -half_height]),
            self.center + np.array([-half_length, half_width, half_height]),
            self.center + np.array([-half_length, -half_width, half_height]),
            self.center + np.array([-half_length, -half_width, -half_height]),
            self.center + np.array([-half_length, half_width, -half_height])
        ]

        self.radius = self.calculate_radius()
        self.collisionSpheres = [self.center]
        
 

@dataclass
class Hitbox(Box):
    graph: nx.Graph
    drone_id: str
    start_pos: any
    end_pos: any
    start_time: any
    duration: any
    simple_duration:float = np.inf
    width: float = configs.buffer_zone_size
    height: float = configs.buffer_zone_size
    edge_center: tuple = field(init=False)
    start_node: any = None
    end_node:any = None
    color:str = 'cyan'
    length = 0.0
    edge_start_pos: any = None
    edge_end_pos: any = None
    edge_duration: float = None
    axes : any = None
    projection : any = None
    radius: any = None
    collisionSpheres: any = None

    def __post_init__(self):
        # Calculate the initial center position and then call the parent class's init
        self.length = euclidean(self.start_pos, self.end_pos)
        center = (np.array(self.start_pos) + np.array(self.end_pos)) / 2
        super().__init__(center, self.length, self.width, self.height)
        self.gen_box()
        
    
    def calculate_radius(self):
        return np.max(np.linalg.norm(self.vertices - self.center, axis=1))

    def gen_box(self)->list:
        start_pos = np.array(self.start_pos)
        end_pos = np.array(self.end_pos)
 
        edge_vector = end_pos - start_pos
        edge_length = np.linalg.norm(edge_vector)

        if edge_length == 0.0:
            raise

        edge_dir = edge_vector / edge_length

        




        # Create perpendicular vectors to define the cuboid's faces
        perp_vector_1 = np.cross(edge_dir, [1, 0, 0])
        if np.linalg.norm(perp_vector_1) == 0:  # handle collinearity
            perp_vector_1 = np.cross(edge_dir, [0, 1, 0])
        perp_vector_1 = perp_vector_1 / np.linalg.norm(perp_vector_1)

        perp_vector_2 = np.cross(edge_dir, perp_vector_1)
        perp_vector_2 = perp_vector_2 / np.linalg.norm(perp_vector_2)

        # Calculate the half dimensions
        half_length = edge_length / 2
        half_width = self.width / 2
        half_height = self.height / 2

        # Compute the edge center and vertices of the cuboid
        self.edge_center = (start_pos + end_pos) / 2
        self.vertices = [
            self.center + half_length * edge_dir + half_width * perp_vector_1 + half_height * perp_vector_2,
            self.center + half_length * edge_dir - half_width * perp_vector_1 + half_height * perp_vector_2,
            self.center + half_length * edge_dir - half_width * perp_vector_1 - half_height * perp_vector_2,
            self.center + half_length * edge_dir + half_width * perp_vector_1 - half_height * perp_vector_2,
            self.center - half_length * edge_dir + half_width * perp_vector_1 + half_height * perp_vector_2,
            self.center - half_length * edge_dir - half_width * perp_vector_1 + half_height * perp_vector_2,
            self.center - half_length * edge_dir - half_width * perp_vector_1 - half_height * perp_vector_2,
            self.center - half_length * edge_dir + half_width * perp_vector_1 - half_height * perp_vector_2
        ]

        self.radius = self.calculate_radius()
        self.collisionSpheres = generate_points_along_line(self.start_pos,self.end_pos)
    
    def split_box(self, fixed_length=configs.box_fixed_length) -> list:
        total_length = self.length
        num_segments = ceil(total_length / fixed_length)
        overlap = (num_segments * fixed_length - total_length) / (num_segments - 1) if num_segments > 1 else 0

        total_duration = self.duration
        segment_duration = total_duration / num_segments

        segments = []
        start_pos = np.array(self.start_pos)
        current_start_time = self.start_time
        for i in range(num_segments):
            end_pos = start_pos + (fixed_length) * (np.array(self.end_pos) - np.array(self.start_pos)) / total_length
            segment_end_time = current_start_time + segment_duration

            if i == num_segments - 1:  # Handle the last segment explicitly
                end_pos = np.array(self.end_pos)
                segment_end_time = self.start_time + self.duration
        
            
            segment = Hitbox(
                graph=self.graph,
                drone_id=self.drone_id,
                start_pos=start_pos.tolist(),
                end_pos=end_pos.tolist(),
                start_time=current_start_time,
                duration=segment_duration,
                edge_start_pos = self.start_pos,
                edge_end_pos = self.end_pos,
                edge_duration = self.duration            
            )

            




            segments.append(segment)

            # Update start_pos and current_start_time for the next segment
            start_pos = end_pos - overlap * (np.array(self.end_pos) - np.array(self.start_pos)) / total_length
            current_start_time = segment_end_time

        return segments
    
class ObstacleBox(Box):
    """ ObstacleBox for static obstacles, inherits from Box. """
    def __init__(self, center, length, width, height):
        super().__init__(center, length, width, height)

    def gen_box(self):
        # Directions along the coordinate axes
        x_dir = np.array([1, 0, 0])
        y_dir = np.array([0, 1, 0])
        z_dir = np.array([0, 0, 1])
        
        half_length = self.length / 2
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Creating vertices for a cuboid
        self.vertices = [
            self.center + half_length * x_dir + half_width * y_dir + half_height * z_dir,
            self.center + half_length * x_dir - half_width * y_dir + half_height * z_dir,
            self.center + half_length * x_dir - half_width * y_dir - half_height * z_dir,
            self.center + half_length * x_dir + half_width * y_dir - half_height * z_dir,
            self.center - half_length * x_dir + half_width * y_dir + half_height * z_dir,
            self.center - half_length * x_dir - half_width * y_dir + half_height * z_dir,
            self.center - half_length * x_dir - half_width * y_dir - half_height * z_dir,
            self.center - half_length * x_dir + half_width * y_dir - half_height * z_dir
        ]

class Building(Box):
    """ ObstacleBox for static obstacles, inherits from Box. """
    def __init__(self, center, length, width, height):
        super().__init__(center, length, width, height)
        self.min_corner = None
        self.max_corner = None
        self.gen_box()  # Automatically generate the box on initialization

    def gen_box(self):
        """ Generate the min and max corners of the box based on its center, length, width, and height. """
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        self.min_corner = self.center - np.array([half_length,half_width, half_height])
        self.max_corner = self.center + np.array([half_length,half_width,half_height])

        


