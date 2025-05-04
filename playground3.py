import numpy as np
from collections import namedtuple
import math

# Define a simple Building container
Building = namedtuple('Building', ['position', 'width', 'depth', 'height'])

def generate_city(bounds,
                  obstacle_density,
                  base_size_pct=0.1,
                  min_height=5,
                  max_height=150):
    """
    Generate a grid-like city of buildings with base size as a percentage of map footprint.

    Guarantees:
      - obstacle_density > 0
      - base_size_pct in (0,1]
      - base_size = base_size_pct * min(map_width, map_depth)
      - 0 <= min_height <= max_height <= map_height
      - building footprints do not overlap
      - heights ∈ [min_height, max_height]
      - final volume ratio matches obstacle_density exactly

    Parameters
    ----------
    bounds : ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        Map extents. Must satisfy (z_max - z_min) >= max_height.
    obstacle_density : float
        Desired building-to-free-space volume ratio (must be > 0).
    base_size_pct : float
        Fraction of the smaller horizontal dimension to use as building base.
        e.g. 0.1 yields base_size = 10% of min(map_width, map_depth).
    min_height : float
        Minimum building height (>= 0).
    max_height : float
        Maximum building height (>= min_height).

    Returns
    -------
    List[Building]
        Buildings satisfying constraints and exactly matching requested density.
    """
    # Unpack bounds
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    map_width = x_max - x_min
    map_depth = y_max - y_min
    map_height = z_max - z_min

    # Validate parameters
    if obstacle_density <= 0:
        raise ValueError("obstacle_density must be > 0")
    if not (0 < base_size_pct <= 1):
        raise ValueError("base_size_pct must be in (0, 1]")
    if min_height < 0 or max_height < 0:
        raise ValueError("min_height and max_height must be >= 0")
    if min_height > max_height:
        raise ValueError("min_height cannot exceed max_height")
    if max_height > map_height:
        raise ValueError(f"max_height ({max_height}) exceeds map height ({map_height})")

    # Compute base_size and grid
    base_size = base_size_pct * min(map_width, map_depth)
    n_x = int(np.floor(map_width / base_size))
    n_y = int(np.floor(map_depth / base_size))
    total_cells = n_x * n_y
    if total_cells < 1:
        raise ValueError("base_size too large: no grid cells available.")

    # Ensure non-overlapping footprints
    cell_w = map_width / n_x
    cell_d = map_depth / n_y
    if base_size > cell_w or base_size > cell_d:
        raise ValueError("Computed base_size leads to overlapping footprints.")

    # Volume calculations
    total_map_volume = map_width * map_depth * map_height
    target_building_volume = obstacle_density / (1 + obstacle_density) * total_map_volume
    footprint_area = base_size * base_size
    target_height_sum = target_building_volume / footprint_area

    # Determine building count range
    m_min = math.ceil(target_height_sum / max_height)
    m_max = math.floor(target_height_sum / min_height) if min_height > 0 else total_cells
    m_min = max(m_min, 1)
    m_max = min(m_max, total_cells)
    if m_min > m_max:
        raise ValueError("Cannot satisfy obstacle_density with given height bounds and base_size.")
    avg_h = (min_height + max_height) / 2.0
    m_est = int(round(target_height_sum / avg_h))
    num_buildings = max(m_min, min(m_est, m_max))

    # Select random cells and sample heights
    cells = np.arange(total_cells)
    chosen = np.random.choice(cells, size=num_buildings, replace=False)
    for _ in range(10):
        raw = np.random.uniform(min_height, max_height, size=num_buildings)
        scaled = raw * (target_height_sum / raw.sum())
        if np.all((scaled >= min_height) & (scaled <= max_height)):
            heights = scaled
            break
    else:
        raise ValueError("Failed to generate building heights within bounds.")

    # Assemble Building list
    buildings = []
    for idx, cell in enumerate(chosen):
        i = cell // n_y
        j = cell % n_y
        cx = x_min + (i + 0.5) * cell_w
        cy = y_min + (j + 0.5) * cell_d
        cz = z_min + heights[idx] / 2.0
        buildings.append(Building([cx, cy, cz], base_size, base_size, float(heights[idx])))

    # Final density assertion
    b_vol = footprint_area * np.sum(heights)
    f_vol = total_map_volume - b_vol
    actual = b_vol / f_vol
    assert np.isclose(actual, obstacle_density, rtol=1e-6), \
        f"Density mismatch: actual {actual:.6f}, target {obstacle_density:.6f}"

    return buildings


def plot_city(buildings, bounds, obstacle_density=None):
    """Plot the city in 3D using Matplotlib bars."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for b in buildings:
        x, y, _ = b.position
        dx, dy, dz = b.width, b.depth, b.height
        ax.bar3d(x - dx/2, y - dy/2, 0, dx, dy, dz,
                 shade=True, color='skyblue', edgecolor='k', alpha=0.8)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(0, bounds[2][1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    title = 'Generated City'
    if obstacle_density is not None:
        title += f' — Density: {obstacle_density:.2f}'
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage
    bounds = ((0, 5_000), (0, 5_000), (0, 150))
    density = 0.036
    city = generate_city(bounds, density, base_size_pct=0.03, min_height=5, max_height=150)
    heights = [b.height for b in city]
    print(f"Generated {len(city)} buildings with base_size={city[0].width:.2f}, heights {min(heights):.2f}-{max(heights):.2f} m.")
    plot_city(city, bounds, obstacle_density=density)
