import matplotlib.pyplot as plt

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
    paths = [(nodes[s]["coords"], nodes[g]["coords"]) for s, g in end_nodes]
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


if __name__ == "__main__":
    # --- Sample data for testing ---
    nodes = {
        0: {"coords": (0, 0)},
        1: {"coords": (5, 5)},
        2: {"coords": (0, 5)},
        3: {"coords": (5, 0)},
        4: {"coords": (2, 6)},
        5: {"coords": (6, 2)},
    }
    end_nodes = [
        (0, 1),  # UAV 0: 0→1
        (2, 3),  # UAV 1: 2→3
        (4, 5),  # UAV 2: 4→5
    ]

    # 1) Compute intersections as list of lists
    intersections = compute_segment_intersections(nodes, end_nodes)
    print("Intersections per UAV:")
    for i, lst in enumerate(intersections):
        print(f" UAV {i}:", lst)

    # 2) Plot segments and their intersections
    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.cm.tab10.colors

    # draw each UAV segment
    for idx, (s_id, g_id) in enumerate(end_nodes):
        x1, y1 = nodes[s_id]["coords"]
        x2, y2 = nodes[g_id]["coords"]
        ax.plot([x1, x2], [y1, y2],
                '-', color=colors[idx % len(colors)], lw=2, label=f'UAV {idx}')

    # mark intersections (plot only once per pair with j>i)
    for i, lst in enumerate(intersections):
        for entry in lst:
            if entry is None:
                continue
            j, px, py = entry
            if j > i:
                ax.scatter(px, py, c='k', marker='x', s=100)
                ax.text(px + 0.1, py + 0.1, f'{i}-{j}', fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('UAV Segment Intersections')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.show()
