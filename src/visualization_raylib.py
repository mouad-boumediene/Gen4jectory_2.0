import pyray as pr
from src import configs
from src.agent import UAV
from src.utils import Box
import networkx as nx
import numpy as np
import time
from src.Box import Endbox
from src.utils import draw_box_between_points
from pyray import *
pr.set_trace_log_level(pr.LOG_WARNING)


def draw_uavs(uavs:list[UAV], graph:nx.Graph):
    cube_mesh = pr.gen_mesh_cube(configs.buffer_zone_size, 
                                 configs.buffer_zone_size, 
                                 configs.buffer_zone_size)

    CUBE = pr.load_model_from_mesh(cube_mesh)
    for uav in uavs:

        position = pr.Vector3(*uav.current_position)

        pr.draw_model(CUBE, 
                    position,
                    0.2, 
                    uav.color)
def draw_collision_spheres(uavs:list[UAV],sim_time, show_only=[]):
    sphere_mesh = pr.gen_mesh_sphere(configs.collision_radius,5,10)

    sphere = pr.load_model_from_mesh(sphere_mesh)
    for i, uav in enumerate(uavs):
        for _, list_hitboxes in enumerate(uav.hit_boxes):
            for _, hit_box in enumerate(list_hitboxes):
                
                if show_only is None:

                    if hit_box.start_time <= sim_time <= hit_box.start_time+hit_box.duration:
                        for pos in hit_box.collisionSpheres:
                            position = pr.Vector3(*pos)

                            pr.draw_model(sphere, 
                                        position,
                                        1, 
                                        pr.color_alpha((0,255,0),0.05))
                else:
                    if i in show_only:
                        if hit_box.start_time <= sim_time <= hit_box.start_time+hit_box.duration:
                            for pos in hit_box.collisionSpheres:
                                position = pr.Vector3(*pos)

                                pr.draw_model(sphere, 
                                            position,
                                            1, 
                                            pr.color_alpha((0,255,0),0.05))
                                
def draw_intersection_cylinders(intersections, collision_radius, cylinder_height=150):
    """
    Draw semi-transparent vertical cylinders at each intersection point along the Z-axis.

    intersections: list of lists; each sublist corresponds to one UAV.
        Each element is either a tuple (uav_id, x, y) or None.
    collision_radius: radius of the cylinder base.
    cylinder_height: height of each cylinder along the Z-axis.
    """
    # Pre-generate cylinder mesh and model once
    mesh_radius = int(round(collision_radius))
    mesh_height = int(round(cylinder_height))
    print(mesh_height, mesh_radius)
    slices = 16  # radial subdivisions
    stacks = 5   # vertical subdivisions
    cyl_mesh = pr.gen_mesh_cylinder(
        mesh_radius,
        mesh_height,
        slices
    )
    cyl_model = pr.load_model_from_mesh(cyl_mesh)

    # Draw each cylinder at the intersection coordinates
    for uav_hits in intersections:
        if not uav_hits:
            continue
        for hit in uav_hits:
            if hit is None:
                continue
            # hit is (uav_id, x, y)
            _, x, y = hit
            # Position cylinder so its base sits on z=0 and it extends along Z
            pos = pr.Vector3(x, y, 0)
            # By default, cylinders are oriented along Y; rotate 90° around X to align with Z
            rot_axis = pr.Vector3(1.0, 0.0, 0.0)
            rot_angle = 90.0  # degrees
            # uniform scaling along each axis
            scale = pr.Vector3(1.0, 1.0, 1.0)
            color = pr.color_alpha(pr.GREEN, 0.05)
            pr.draw_model_ex(
                cyl_model,
                pos,
                rot_axis,
                rot_angle,
                scale,
                color
            )


def draw_paths(uavs:list[UAV], graph:nx.Graph):
    cube_mesh = pr.gen_mesh_cube(1,1,1)
    CUBE = pr.load_model_from_mesh(cube_mesh)
    
    for uav in uavs:
        if uav.path is None:
            return
        for i in range(len(uav.path)):
            position = graph.nodes[uav.path[i]]['coords']

            pr.draw_model(CUBE, 
                        position,
                        0.1, 
                        uav.color)
            if i > 0:
                pr.draw_line_3d(graph.nodes[uav.path[i]]['coords'], 
                                graph.nodes[uav.path[i-1]]['coords'],pr.color_alpha((255,0,0),0.5) )
                
                


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def calculate_rotation_matrix(A, B):
    # Calculate the direction vector
    direction = np.array(B) - np.array(A)
    
    # Normalize the direction vector
    
    forward = direction / np.linalg.norm(direction)
    
    # Define an arbitrary up vector (commonly (0, 1, 0))
    up_world = np.array([0, 1, 0], dtype=float)
    
    # Calculate the side vector (right direction)
    side = np.cross(up_world, forward)
    side = side / np.linalg.norm(side)
    
    # Recalculate the up vector to ensure orthogonality
    up = np.cross(forward, side)
    
    # Create the rotation matrix
    rotation_matrix = np.array([side, up, forward])
    
    return rotation_matrix.T  # Transpose to align with the standard rotation matrix format
def get_yaw_pitch_roll(p1, p2):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Calculate the midpoint
    midpoint = (p1 + p2) / 2
    
    # Translate points to have the midpoint at the origin
    p1_translated = p1 - midpoint
    p2_translated = p2 - midpoint
    
    # Calculate direction vector
    direction_vector = p2_translated - p1_translated
    direction_norm = np.linalg.norm(direction_vector)
    
    if direction_norm == 0:
        raise ValueError("Direction vector is zero-length, cannot calculate orientation.")
    
    # Normalize direction vector
    normalized_direction = direction_vector / direction_norm
    
    # Calculate yaw, pitch, and roll
    yaw = np.arctan2(normalized_direction[2], normalized_direction[0])  # yaw in radians
    pitch = np.arcsin(normalized_direction[1])  # pitch in radians
    roll = 0.0  # Assuming no roll, or calculate based on additional requirements
    
    # Convert to degrees
    yaw_degrees = np.degrees(yaw)
    pitch_degrees = np.degrees(pitch)
    roll_degrees = np.degrees(roll)
    
    return yaw_degrees, pitch_degrees, roll_degrees

def draw_box(vertices, color):
    # Ensure vertices are pr.Vector3 objects
    vertices = [pr.Vector3(vertex[0], vertex[1], vertex[2]) for vertex in vertices]
    
    # Top face
    pr.draw_triangle_3d(vertices[0], vertices[1], vertices[2], color)
    pr.draw_triangle_3d(vertices[0], vertices[2], vertices[3], color)

    # Bottom face
    pr.draw_triangle_3d(vertices[4], vertices[5], vertices[6], color)
    pr.draw_triangle_3d(vertices[4], vertices[6], vertices[7], color)

    # Front face
    pr.draw_triangle_3d(vertices[0], vertices[1], vertices[5], color)
    pr.draw_triangle_3d(vertices[0], vertices[5], vertices[4], color)

    # Back face
    pr.draw_triangle_3d(vertices[2], vertices[3], vertices[7], color)
    pr.draw_triangle_3d(vertices[2], vertices[7], vertices[6], color)

    # Left face
    pr.draw_triangle_3d(vertices[1], vertices[2], vertices[6], color)
    pr.draw_triangle_3d(vertices[1], vertices[6], vertices[5], color)

    # Right face
    pr.draw_triangle_3d(vertices[0], vertices[3], vertices[7], color)
    pr.draw_triangle_3d(vertices[0], vertices[7], vertices[4], color)

def draw_boxes(uavs:list[UAV], sim_time:float, show_only=None, draw_endboxes =False):
    for i, uav in enumerate(uavs):
        for _, list_hitboxes in enumerate(uav.hit_boxes):
            for _, hit_box in enumerate(list_hitboxes):
                if not draw_endboxes and isinstance(hit_box,Endbox):
                    continue
                if show_only is None:
                    if hit_box.start_time <= sim_time <= hit_box.start_time+hit_box.duration:
                        
                        draw_box_between_points(hit_box.vertices)
                else:
                    if i in show_only:
                        if hit_box.start_time <= sim_time <= hit_box.start_time+hit_box.duration:
                            draw_box_between_points(hit_box.vertices)





def update_uavs_poses(uavs: list[UAV], end_nodes, graph: nx.Graph, delta_time: float, sim_time:float, clock_scen:bool):

    all_reached_goal = True
    for i,uav in enumerate(uavs):
        start, goal = end_nodes[i]
        if uav.path == None:
            uav.start_position = np.array(graph.nodes[start]['coords'])
            uav.goal_position = np.array(graph.nodes[goal]['coords'])
            uav.current_position = uav.start_position
            pr.draw_line_3d(pr.Vector3(*uav.start_position),pr.Vector3(*uav.goal_position),pr.GREEN)
            continue
        if type(uav.start_position) != np.ndarray:
            uav.start_position = np.array(graph.nodes[start]['coords'])
            uav.goal_position = np.array(graph.nodes[goal]['coords'])
            pr.draw_sphere(pr.Vector3(*uav.goal_position),20,pr.GREEN)
            pr.draw_sphere(pr.Vector3(*uav.start_position),20,pr.GREEN)

            uav.current_position = np.array(uav.start_position)
        if sim_time < uav.start_time:
            uav.current_position = uav.start_position
            continue

        if uav.current_waypoint_index < len(uav.path) - 1:
            all_reached_goal = False
            start_point = np.array(graph.nodes[uav.path[uav.current_waypoint_index]]['coords'])
            end_point = np.array(graph.nodes[uav.path[uav.current_waypoint_index + 1]]['coords'])
            segment_distance = np.linalg.norm(end_point - start_point)
            if clock_scen:
                segment_time = uav.hit_boxes[uav.current_waypoint_index][-1].edge_duration
            else:
                segment_time = uav.hit_boxes[uav.current_waypoint_index+1][-1].edge_duration

            
            speed = segment_distance / segment_time  # Speed in units per second
            #print(segment_distance,segment_time, speed)

            # Update the interpolation factor based on speed and delta_time
            uav.t += speed * delta_time / segment_distance
            
            if uav.t >= 1.0:
                # Move to the next waypoint
                uav.t = 0
                uav.current_waypoint_index += 1
                uav.current_position = end_point
            else:
                # Interpolate the position
                uav.current_position = start_point + uav.t * (end_point - start_point)
                uav.update_position(uav.current_position)
        else:
            # The UAV has reached the final waypoint, but we continue to check others
            continue
    
    # If all UAVs have reached their goals, reset their positions
    if all_reached_goal:
        for uav in uavs:
            uav.reset_position()
        sim_time = 0.0
    return sim_time

def draw_border(bounds=configs.bounds):
    # Calculate the sizes
    size_x = bounds[0][1] - bounds[0][0]
    size_y = bounds[1][1] - bounds[1][0]
    size_z = bounds[2][1] - bounds[2][0]

    # Define the 8 vertices of the cube with the lower-left vertex at (0, 0, 0)
    vertices = [
        (0, 0, 0),
        (size_x, 0, 0),
        (size_x, size_y, 0),
        (0, size_y, 0),
        (0, 0, size_z),
        (size_x, 0, size_z),
        (size_x, size_y, size_z),
        (0, size_y, size_z),
    ]

    # Define the 12 edges of the cube
    edges = [
        (vertices[0], vertices[1]),
        (vertices[1], vertices[2]),
        (vertices[2], vertices[3]),
        (vertices[3], vertices[0]),
        (vertices[4], vertices[5]),
        (vertices[5], vertices[6]),
        (vertices[6], vertices[7]),
        (vertices[7], vertices[4]),
        (vertices[0], vertices[4]),
        (vertices[1], vertices[5]),
        (vertices[2], vertices[6]),
        (vertices[3], vertices[7]),
    ]

    # Draw the edges
    for edge in edges:
        pr.draw_line_3d(edge[0], edge[1], pr.GRAY)


def draw_graph(graph, node_color=pr.RED, edge_color=pr.GRAY, node_size=0.1):
    # Extract positions from the graph's 'coords' attribute
    positions = nx.get_node_attributes(graph, 'coords')

    # Draw edges
    for edge in graph.edges():
        start_pos = positions[edge[0]]
        end_pos = positions[edge[1]]
        pr.draw_line_3d(start_pos, end_pos, pr.color_alpha(pr.GRAY,0.1))

    """ # Draw nodes
    for _, pos in positions.items():
        pr.draw_sphere(pos, node_size, node_color) """


def draw_axis(bounds=configs.bounds):
    # Draw the axis
    axis_length = 10.0
    #pr.draw_line_3d(pr.Vector3(0, 0, 0), pr.Vector3(axis_length, 0, 0), pr.RED)    # X-axis (red)
    #pr.draw_line_3d(pr.Vector3(0, 0, 0), pr.Vector3(0, axis_length, 0), pr.GREEN)  # Y-axis (green)
    #pr.draw_line_3d(pr.Vector3(0, 0, 0), pr.Vector3(0, 0, axis_length), pr.BLUE)   # Z-axis (blue)
    # Draw three boxes, one taller in each axis
    box_size = bounds[0][1]/200
    box_length = bounds[0][1]/10

    # Box taller in the X-axis
    pr.draw_cube(pr.Vector3(bounds[0][1]-box_length/2, bounds[0][1], 0), box_length, box_size, box_size, pr.RED)

    # Box taller in the Y-axis
    pr.draw_cube(pr.Vector3(bounds[0][1], bounds[0][1]-box_length/2, 0), box_size, box_length, box_size, pr.GREEN)

    # Box taller in the Z-axis
    pr.draw_cube(pr.Vector3(bounds[0][1], bounds[0][1], box_length/2), box_size, box_size, box_length, pr.BLUE)



def draw_buildings(buildings):
    for building in buildings:
        pr.draw_cube(pr.Vector3(*building.center), building.length, building.width, building.height, pr.LIGHTGRAY)
        pr.draw_cube_wires(pr.Vector3(*building.center), building.length, building.width, building.height, pr.GRAY)

def draw_ground(bounds, thickness=1.0,
                color=pr.LIGHTGRAY, wire_color=pr.GRAY):
    """
    Draws a flat ground plane as a thin cube.

    bounds: [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
      x,y are your horizontal extents; z is up.
    thickness: how “tall” the ground‐cube is in the up (z) direction.
    """
    (xmin, xmax), (ymin, ymax), _ = bounds

    # half‐sizes
    size_x = xmax - xmin
    size_y = ymax - ymin
    half_thick = -thickness / 2.0

    # center of the ground cube:
    #   x: midpoint of [xmin,xmax]
    #   y: midpoint of [ymin,ymax]
    #   z: sit it so its top is at z=0 (i.e. bottom at z=-thickness)
    center = pr.Vector3(
        xmin + size_x/2.0,
        ymin + size_y/2.0,
        -half_thick
    )

    # draw a flat cube (width=X, depth=Y, height=thickness)
    pr.draw_cube(       center, size_x, size_y, thickness, color)
    pr.draw_cube_wires(center, size_x, size_y, thickness, wire_color)

def animate_raylib(uavs: list[UAV], buildings: list[Box], graph=None, bounds=False, end_nodes=[], show_only =None, clock_scen = False, draw_endboxes = False, intersections=[None]):

    pr.init_window(configs.window_w, configs.window_h, "Visualizing 4D drone path planning")
    pr.set_target_fps(configs.fps)
    
    cam_speed = 50.0
    sim_time = 0 
    last_time = time.time()
    paused = False
    # Calculate the center of the box
    box_center = [np.mean(axis) for axis in bounds]  # [10.0, 10.0, 10.0]
    cam = pr.Camera3D([bounds[0][1]*1.1, bounds[1][1]*1.1, bounds[2][1]*1.1], box_center, [0.0, 0.0, 500.0], 45.0, 0)

    while not pr.window_should_close():
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        
        dt = pr.get_frame_time()
        pr.update_camera(cam, pr.CameraMode.CAMERA_FREE)

        # compute forward & right axes based on current camera orientation
        forward = pr.vector3_normalize(pr.vector3_subtract(cam.target, cam.position))
        right   = pr.vector3_normalize(pr.vector3_cross_product(forward, pr.Vector3(0, 1, 0)))

        # WSAD controls
        if pr.is_key_down(pr.KEY_W):
            if pr.is_key_down(pr.KEY_LEFT_SHIFT):
                delta = pr.vector3_scale(forward,  cam_speed * dt)
            else:
                delta = pr.vector3_scale(forward,  cam_speed * dt * 0.1)
            cam.position = pr.vector3_add(cam.position, delta)
            cam.target   = pr.vector3_add(cam.target,   delta)
        if pr.is_key_down(pr.KEY_S):
            if pr.is_key_down(pr.KEY_LEFT_SHIFT):
                delta = pr.vector3_scale(forward,  -cam_speed * dt)
            else:
                delta = pr.vector3_scale(forward,  -cam_speed * dt * 0.1)
            cam.position = pr.vector3_add(cam.position, delta)
            cam.target   = pr.vector3_add(cam.target,   delta)
        if pr.is_key_down(pr.KEY_D):
            if pr.is_key_down(pr.KEY_LEFT_SHIFT):
                delta = pr.vector3_scale(right,  -cam_speed * dt)
            else:
                delta = pr.vector3_scale(right,  -cam_speed * dt * 0.1)
            cam.position = pr.vector3_add(cam.position, delta)
            cam.target   = pr.vector3_add(cam.target,   delta)
        if pr.is_key_down(pr.KEY_A):
            if pr.is_key_down(pr.KEY_LEFT_SHIFT):
                delta = pr.vector3_scale(right,  cam_speed * dt)
            else:
                delta = pr.vector3_scale(right,  cam_speed * dt * 0.1)
            cam.position = pr.vector3_add(cam.position, delta)
            cam.target   = pr.vector3_add(cam.target,   delta)

        # Q/E for vertical movement
        if pr.is_key_down(pr.KEY_SPACE):
            cam.position.z += cam_speed * dt
            cam.target.z   += cam_speed * dt
        if pr.is_key_down(pr.KEY_LEFT_CONTROL):
            cam.position.z -= cam_speed * dt
            cam.target.z  -= cam_speed * dt
        
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        
        pr.begin_mode_3d(cam)
        #cam.position = pr.Vector3(0,0,200)
        #pr.draw_grid(20, 1.0)
        draw_axis(bounds)
        draw_buildings(buildings)
        draw_ground(bounds, thickness=0.05, color=(245,245,245), wire_color=pr.GRAY)



        if not paused:
            sim_time = update_uavs_poses(uavs,end_nodes, graph, delta_time, sim_time, clock_scen)
            sim_time += delta_time
        draw_border(bounds)
        #draw_graph(graph)
        draw_uavs(uavs, graph)
        #draw_collision_spheres(uavs,sim_time,show_only)
        #draw_intersection_cylinders(intersections, configs.intersection_collision_radius)
        draw_paths(uavs, graph)
        draw_boxes(uavs, sim_time, show_only,draw_endboxes)
        #draw_mesh_boxes(uavs, sim_time)

        pr.end_mode_3d()
        pr.end_drawing()

        # Check for pause/unpause
        if pr.is_key_pressed(pr.KEY_P):
            paused = not paused
        if pr.is_key_pressed(pr.KEY_T):
            pr.take_screenshot('Gen4jectory_viz.png')

    pr.close_window()

if __name__ == "__main__":
    # Ensure you pass the required UAV and obstacle objects, graph, and any other necessary arguments to animate_raylib()
    pass#animate_raylib(uavs, obstacles)