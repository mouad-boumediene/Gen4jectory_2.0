import numpy as np
from math import ceil, sqrt
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D

class Hitbox:
    def __init__(
        self, graph, drone_id: int,
        start_pos: List[float], end_pos: List[float],
        start_time: float, duration: float,
        edge_start_pos: List[float], edge_end_pos: List[float], edge_duration: float
    ):
        # Store original path segment properties
        self.graph = graph
        self.drone_id = drone_id
        self.start_pos = start_pos  # [x, y]
        self.end_pos = end_pos      # [x, y]
        self.start_time = start_time
        self.duration = duration
        # Context of full edge
        self.edge_start_pos = edge_start_pos
        self.edge_end_pos = edge_end_pos
        self.edge_duration = edge_duration

    def split_box(
        self,
        intersections: List[Tuple[int, float, float]],
        radius: float,
        fixed_length: float
    ) -> List['Hitbox']:
        """
        Split path segment into fine parts inside circles around 2D intersection points,
        coarse elsewhere. Uses `radius` both to detect proximity and to draw circles.
        """
        start = np.array(self.start_pos)
        end = np.array(self.end_pos)
        vec = end - start
        total_length = np.linalg.norm(vec)
        total_time = self.duration

        # Compute entry/exit t for each intersection circle
        t_points: List[float] = []
        for _, xi, yi in intersections:
            p = np.array([xi, yi])
            proj_dist = np.dot(p - start, vec) / total_length
            closest_pt = start + vec * (proj_dist / total_length)
            d_perp = np.linalg.norm(closest_pt - p)
            if d_perp > radius:
                continue
            delta = sqrt(radius**2 - d_perp**2)
            t_entry = max(0.0, min(1.0, (proj_dist - delta) / total_length))
            t_exit  = max(0.0, min(1.0, (proj_dist + delta) / total_length))
            t_points.extend([t_entry, t_exit])

        breakpoints = sorted({0.0, *t_points, 1.0})
        segments: List[Hitbox] = []
        current_time = self.start_time

        # Split by interval and refine if inside any circle
        for i in range(len(breakpoints) - 1):
            t0, t1 = breakpoints[i], breakpoints[i+1]
            seg_len = (t1 - t0) * total_length
            seg_time = (t1 - t0) * total_time
            seg_start = start + vec * t0
            seg_end = start + vec * t1

            mid_t = 0.5*(t0 + t1)
            mid_pt = start + vec * mid_t
            inside = any(
                np.linalg.norm(mid_pt - np.array([xi, yi])) <= radius
                for _, xi, yi in intersections
            )

            if inside and seg_len > fixed_length:
                # subdivide finely
                n_sub = ceil(seg_len / fixed_length)
                overlap = ((n_sub*fixed_length - seg_len)/(n_sub-1)) if n_sub>1 else 0
                dt = seg_time / n_sub
                s_pos = seg_start.copy()
                for _ in range(n_sub):
                    e_pos = s_pos + (fixed_length/seg_len)*(seg_end - seg_start)
                    segments.append(Hitbox(
                        graph=self.graph,
                        drone_id=self.drone_id,
                        start_pos=s_pos.tolist(),
                        end_pos=e_pos.tolist(),
                        start_time=current_time,
                        duration=dt,
                        edge_start_pos=self.start_pos,
                        edge_end_pos=self.end_pos,
                        edge_duration=self.duration
                    ))
                    s_pos = e_pos - overlap*(seg_end - seg_start)/seg_len
                    current_time += dt
            else:
                # coarse segment
                segments.append(Hitbox(
                    graph=self.graph,
                    drone_id=self.drone_id,
                    start_pos=seg_start.tolist(),
                    end_pos=seg_end.tolist(),
                    start_time=current_time,
                    duration=seg_time,
                    edge_start_pos=self.start_pos,
                    edge_end_pos=self.end_pos,
                    edge_duration=self.duration
                ))
                current_time += seg_time

        return segments


def draw_box_2d(
    ax,
    seg: Hitbox,
    width: float,
    color: str
):
    """
    Draw the 2D projection of a 3D box as a rectangle on XY plane.
    """
    s = np.array(seg.start_pos)
    e = np.array(seg.end_pos)
    d = e - s
    d_unit = d / np.linalg.norm(d)
    perp = np.array([-d_unit[1], d_unit[0]]) * (width/2)

    # rectangle corners
    c0 = s + perp
    c1 = s - perp
    c2 = e - perp
    c3 = e + perp
    rect = np.vstack([c0, c1, c2, c3])

    poly = Polygon(rect, closed=True, edgecolor=color, facecolor=color, alpha=0.3)
    ax.add_patch(poly)
    ax.plot([s[0], e[0]], [s[1], e[1]], color=color, linewidth=2)


def visualize_hitbox_split_2d(
    hitbox: Hitbox,
    intersections: List[Tuple[int, float, float]],
    radius: float,
    fixed_length: float,
    box_width: float = 1000,
    sphere_color: str = 'green',
    sphere_alpha: float = 0.2
):
    """
    2D visualization: projected rectangles for hitbox segments and circles
    for intersection zones, using `radius` for both proximity and drawing.
    """
    segments = hitbox.split_box(intersections, radius, fixed_length)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')

    # Draw intersection circles
    for _, xi, yi in intersections:
        circ = Circle((xi, yi), radius,
                      edgecolor=sphere_color, facecolor=sphere_color,
                      alpha=sphere_alpha, linestyle='--')
        ax.add_patch(circ)
        ax.plot(xi, yi, 'o', color=sphere_color)

    # Draw boxes
    for seg in segments:
        length = np.linalg.norm(np.array(seg.end_pos) - np.array(seg.start_pos))
        color = 'red' if length <= fixed_length+1e-6 else 'blue'
        draw_box_2d(ax, seg, box_width, color)

    # Legend
    legend_elems = [
        Line2D([0], [0], color='red', lw=4, label='Fine Segment'),
        Line2D([0], [0], color='blue', lw=4, label='Coarse Segment'),
        Line2D([0], [0], marker='o', color=sphere_color, linestyle='', markersize=8, label='Intersection Center'),
        Circle((0,0), radius, edgecolor=sphere_color, facecolor=sphere_color, alpha=sphere_alpha, label='Intersection Zone')
    ]
    ax.legend(handles=legend_elems)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('2D Projection: Hitbox Segments & Intersection Zones')
    plt.show()

if __name__ == '__main__':
    sample_intersections = [
        (2, 50_000, 40_000),
    ]
    hb = Hitbox(
        graph=None, drone_id=1,
        start_pos=[0,0], end_pos=[134000,115000],
        start_time=0.0, duration=10.0,
        edge_start_pos=[98000,96000], edge_end_pos=[134000,115000], edge_duration=10.0
    )
    visualize_hitbox_split_2d(
        hb,
        sample_intersections,
        radius=20_000,
        fixed_length=5_000,
        box_width=5000,
        sphere_color='green',
        sphere_alpha=0.3
    )
