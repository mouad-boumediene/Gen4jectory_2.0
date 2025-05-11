"""
Module: Box.py
Description:
    - Box (ABC): oriented‐bounding‐box base class with SAT collision test
    - Endbox / Hitbox: specialized boxes for UAV hit‐zones, with segmentation
    - Buildings: static obstacle representations in 3D space
"""
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
        self.axes       = np.eye(3)                 # shape (3,3)
        self.half_sizes = np.array([length/2,
                                    width/2,
                                    height/2])
        self.gen_box()
    
    @abstractmethod
    def gen_box(self):
        """ Method to generate the vertices of the box. """
        pass

    def collides_with(self, other: "Box",margin = 0.6) -> bool:
        """
        Exact OBB‐vs‐OBB collision test (SAT) for ANY two Boxes, with an optional safety margin.

        Parameters:
        -----------
        other : Box
            The other box to test against.
        margin : float, optional
            How much to “grow” each box’s half‐extents in all directions
            (default is 0.0, i.e. no inflation).

        Requires each box to have:
          - .axes       : (3,3) array of local unit‐axes
          - .half_sizes : (3,) array of half‐extents along those axes
          - .center     : (3,) center point

        Returns:
        --------
        bool
            True if the (inflated) boxes overlap; False otherwise.
        """
        
        A = self.axes         # (3,3)
        B = other.axes        # (3,3)

        # Inflate half‐sizes by margin
        a = self.half_sizes + margin  # (3,)
        b = other.half_sizes + margin

        # 1) Rotation matrix R_ij = A_i · B_j
        R = A @ B.T

        # 2) Translation vector t in A’s local frame
        t = A @ (other.center - self.center)

        # 3) Absolute + epsilon for stability
        absR = np.abs(R) + 1e-8

        # 4) A’s face‐normals
        for i in range(3):
            if abs(t[i]) > a[i] + np.dot(b, absR[i]):
                return False

        # 5) B’s face‐normals
        for j in range(3):
            if abs(t @ R[:, j]) > np.dot(a, absR[:, j]) + b[j]:
                return False

        # 6) cross‐product axes A_i × B_j
        for i in range(3):
            for j in range(3):
                ra = (a[(i+1)%3] * absR[(i+2)%3, j] +
                      a[(i+2)%3] * absR[(i+1)%3, j])
                rb = (b[(j+1)%3] * absR[i, (j+2)%3] +
                      b[(j+2)%3] * absR[i, (j+1)%3])
                lhs = abs(t[(i+2)%3] * R[(i+1)%3, j] -
                          t[(i+1)%3] * R[(i+2)%3, j])
                if lhs > ra + rb:
                    return False

        return True

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
        self.axes       = np.eye(3)
        self.half_sizes = np.array([half_length,
                                     half_width,
                                     half_height])


    
        
 

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
        # record your local axes (for Endbox it's just the world axes)
        self.axes       = np.eye(3)                 # shape (3,3)
        self.half_sizes = np.array([self.length/2,
                                    self.width/2,
                                    self.height/2])
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

        self.axes       = np.stack([edge_dir,
                             perp_vector_1,
                             perp_vector_2], axis=0)  # shape (3,3)
        self.half_sizes = np.array([half_length,
                                    half_width,
                                    half_height])
        
    
    def split_box_new(self,
                  sphere_centers: list,
                  sphere_radius: float,
                  coarse_length: float = configs.box_fixed_length * 100,
                  fine_length: float = configs.box_fixed_length) -> list:
        """
        Splits a 3D hitbox into overlapping fixed-length segments in both outside and inside regions,
        while preserving exactly computed times even if spatial endpoints are clamped.

        Args:
            sphere_centers: list of (drone_id, x, y) tuples or None entries
            sphere_radius:  radius of cylinders in the XY plane
            coarse_length: target segment length for outside regions
            fine_length:   target segment length for inside regions

        Returns:
            List of Hitbox instances covering [self.start_pos, self.end_pos].
        """
        # 0) Normalize sphere_centers to XY‐plane points
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

        D_unit = D / total_len

        # 2) Project to XY for intersection math
        P0_xy = P0[:2]
        P1_xy = P1[:2]
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
                    # Possible full containment
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

        # Merge overlapping inside intervals
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

        # 5) Build raw segment data (store times BEFORE any spatial clamp)
        segments = []

        def add_segment(p0, p1, region_label):
            # Compute fractional positions
            t0 = np.linalg.norm(p0 - P0) / total_len
            t1 = np.linalg.norm(p1 - P0) / total_len
            # Map to absolute times
            start_tm = self.start_time + t0 * self.duration
            seg_dur  = (t1 - t0) * self.duration
            segments.append({
                "p0": p0,
                "p1": p1,
                "region": region_label,
                "t0": t0,
                "t1": t1,
                "start_time": start_tm,
                "duration": seg_dur
            })

        # 5a) Outside: overlapping segments of length coarse_length
        for a, b in outside:
            P0_o = P0 + a * D
            P1_o = P0 + b * D
            region_len_o = np.linalg.norm(P1_o - P0_o)
            if region_len_o <= 0:
                continue

            count_o = ceil(region_len_o / coarse_length)
            overlap_o = ((count_o * coarse_length - region_len_o) / (count_o - 1)
                        if count_o > 1 else 0.0)
            dir_o = (P1_o - P0_o) / region_len_o
            start_o = P0_o

            for i in range(count_o):
                end_o = start_o + dir_o * coarse_length
                if i == count_o - 1:
                    end_o = P1_o
                add_segment(start_o, end_o, 'outside')
                start_o = end_o - dir_o * overlap_o

        # 5b) Inside: overlapping segments of length fine_length
        for a, b in inside:
            P0_i = P0 + a * D
            P1_i = P0 + b * D
            region_len_i = np.linalg.norm(P1_i - P0_i)
            if region_len_i <= 0:
                continue

            count_i = ceil(region_len_i / fine_length)
            overlap_i = ((count_i * fine_length - region_len_i) / (count_i - 1)
                        if count_i > 1 else 0.0)
            dir_i = (P1_i - P0_i) / region_len_i
            start_i = P0_i

            for j in range(count_i):
                end_i = start_i + dir_i * fine_length
                if j == count_i - 1:
                    end_i = P1_i
                add_segment(start_i, end_i, 'inside')
                start_i = end_i - dir_i * overlap_i

        # 6) Spatial clamp of endpoints only (leave stored times intact)
        eps = 1e-8
        segments.sort(key=lambda s: np.dot(s["p0"] - P0, D_unit))
        if segments:
            first = segments[0]
            if first["t0"] > eps:
                first["p0"] = P0
            last = segments[-1]
            if (1.0 - last["t1"]) > eps:
                last["p1"] = P1

        # 7) Instantiate Hitbox objects using stored temporal info
        hitboxes = []
        for seg in segments:
            hb = Hitbox(
                graph=self.graph,
                drone_id=self.drone_id,
                start_pos=seg["p0"].tolist(),
                end_pos=seg["p1"].tolist(),
                start_time=seg["start_time"],
                duration=seg["duration"],
                edge_start_pos=self.start_pos,
                edge_end_pos=self.end_pos,
                edge_duration=self.duration
            )
            hitboxes.append(hb)

        return hitboxes


class Building(Box):
    """ static obstacles, inherits from Box. """
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

        


