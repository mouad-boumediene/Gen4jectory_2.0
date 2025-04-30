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
    
    def split_box_new(self,
              sphere_centers: list,
              sphere_radius: float,
              coarse_length: float = configs.box_fixed_length,
              fine_length:   float = configs.box_fixed_length) -> list:
        """
        Splits a 3D hitbox into fixed-length segments:
        - Each outside‐sphere interval yields one segment spanning from entry to exit.
        - Inside sphere regions use overlapping segments of exact fine_length.

        Args:
        sphere_centers: list of (drone_id, x, y) tuples or None/[None]
        sphere_radius:  radius of circles in XY plane
        coarse_length: unused for outside (intervals not subdivided)
        fine_length:   length of inside segments
        Returns:
        List of Hitbox instances covering [self.start_pos, self.end_pos].
        """
        # 0) Normalize sphere_centers to XY-plane points
        valid_xy = []
        for entry in sphere_centers:
            if not entry or (isinstance(entry, (list, tuple)) and entry[0] is None):
                continue
            _, x, y = entry
            valid_xy.append(np.array((x, y), dtype=float))

        # 1) Parameterize 3D line
        P0 = np.array(self.start_pos, dtype=float)
        P1 = np.array(self.end_pos,   dtype=float)
        D  = P1 - P0
        total_len = np.linalg.norm(D)
        if total_len == 0:
            return []

        # 2) Project to XY for intersection math
        P0_xy = P0[:2]; P1_xy = P1[:2]
        D_xy  = P1_xy - P0_xy

        # 3) Find inside‐sphere t‐intervals in [0,1]
        intervals = []
        if np.linalg.norm(D_xy) > 0 and valid_xy:
            a2 = D_xy.dot(D_xy)
            for C in valid_xy:
                b2 = 2 * D_xy.dot(P0_xy - C)
                c2 = (P0_xy - C).dot(P0_xy - C) - sphere_radius**2
                disc = b2*b2 - 4*a2*c2
                if disc < 0:
                    # potential full containment
                    mid = P0_xy + 0.5 * D_xy
                    if np.linalg.norm(mid - C) <= sphere_radius:
                        intervals.append((0.0, 1.0))
                    continue
                t1 = (-b2 - np.sqrt(disc)) / (2*a2)
                t2 = (-b2 + np.sqrt(disc)) / (2*a2)
                t0, t3 = sorted((t1, t2))
                start, end = max(0.0, t0), min(1.0, t3)
                if end > start:
                    intervals.append((start, end))

        # merge overlapping inside intervals
        intervals.sort(key=lambda x: x[0])
        inside = []
        for a, b in intervals:
            if not inside or a > inside[-1][1]:
                inside.append([a, b])
            else:
                inside[-1][1] = max(inside[-1][1], b)

        # 4) Compute outside = complement on [0,1]
        outside = []
        prev = 0.0
        for a, b in inside:
            if a > prev:
                outside.append((prev, a))
            prev = b
        if prev < 1.0:
            outside.append((prev, 1.0))

        # 5) Build segments list
        segments = []

        # 5a) Outside: one segment per interval
        for a, b in outside:
            p0 = P0 + a * D
            p1 = P0 + b * D
            segments.append((p0, p1, 'outside'))

        # 5b) Inside: overlapping fixed-length segments
        for a, b in inside:
            P0_i = P0 + a * D
            P1_i = P0 + b * D
            region_len = np.linalg.norm(P1_i - P0_i)
            if region_len <= 0:
                continue
            count = ceil(region_len / fine_length)
            overlap = (count * fine_length - region_len) / (count - 1) if count > 1 else 0
            dir_unit = (P1_i - P0_i) / region_len
            start_pt = P0_i
            for i in range(count):
                end_pt = start_pt + dir_unit * fine_length
                if i == count - 1:
                    end_pt = P1_i
                segments.append((start_pt, end_pt, 'inside'))
                start_pt = end_pt - dir_unit * overlap

        # 6) Clamp global endpoints and sort
        D_unit = D / total_len
        segments.sort(key=lambda s: np.dot(s[0] - P0, D_unit))
        if segments:
            _, end0, r0 = segments[0]
            segments[0] = (P0, end0, r0)
            startN, _, rN = segments[-1]
            segments[-1] = (startN, P1, rN)

        # 7) Instantiate Hitbox objects
        hitboxes = []
        for start_pt, end_pt, region in segments:
            t0 = np.linalg.norm(start_pt - P0) / total_len
            t1 = np.linalg.norm(end_pt   - P0) / total_len
            seg_dur = (t1 - t0) * self.duration
            start_tm = self.start_time + t0 * self.duration
            hb = Hitbox(
                graph=self.graph,
                drone_id=self.drone_id,
                start_pos=start_pt.tolist(),
                end_pos=end_pt.tolist(),
                start_time=start_tm,
                duration=seg_dur,
                edge_start_pos=self.start_pos,
                edge_end_pos=self.end_pos,
                edge_duration=self.duration
            )
            hitboxes.append(hb)

        return hitboxes

    
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

        


