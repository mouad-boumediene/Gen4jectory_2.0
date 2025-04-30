import pygame
import numpy as np
from math import ceil

# ---------- Configuration ----------
WIDTH, HEIGHT = 800, 400
LINE_START = np.array([50.0, 350.0])
LINE_END   = np.array([750.0,  50.0])

# Two segment lengths
# Outside intervals are not subdivided
FINE_LEN   =  40.0  # inside circles

# Initial circle centers (2D tuples) and common radius
circle_centers = [
    np.array([200.0, 200.0]),
    np.array([400.0, 150.0]),
    np.array([600.0, 250.0])
]
CIRCLE_RADIUS = 80.0

# Marker settings
START_MARKER_OFFSET = 8.0
END_MARKER_OFFSET   = 16.0
START_MARKER_COLOR  = (0,200,0)
END_MARKER_COLOR    = (200,0,200)
MARKER_RADIUS       = 6
HIGHLIGHT_COLOR     = (255,165,0)
print_now = True

# ---------- Utility: Print Table ----------
def print_table(segments):
    w = [5,16,16,8,8]
    sep = '+' + '+'.join('-'*wi for wi in w) + '+'
    hdr = '|{:^5}|{:^16}|{:^16}|{:^8}|{:^8}|'.format('Idx','Start','End','Len','Reg')
    print(sep); print(hdr); print(sep)
    for i,(p0,p1,reg) in enumerate(segments):
        x0,y0 = p0; x1,y1 = p1
        length = np.linalg.norm(p1-p0)
        row = '|{:>5}|({:6.1f},{:6.1f})|({:6.1f},{:6.1f})|{:8.1f}|{:8}|'.format(
            i, x0,y0, x1,y1, length, reg)
        print(row)
    print(sep)

# ---------- Line Split Function ----------
def split_line_2d(P0, P1, centers, radius, fine_len):
    D = P1 - P0
    L = np.linalg.norm(D)
    if L == 0:
        return []
    # Find inside intervals in t-space
    intervals = []
    for C in centers:
        a = D.dot(D)
        b = 2 * D.dot(P0 - C)
        c = (P0 - C).dot(P0 - C) - radius*radius
        disc = b*b - 4*a*c
        if disc < 0:
            mid = (P0 + P1) / 2
            if np.linalg.norm(mid - C) <= radius:
                intervals.append((0.0,1.0))
            continue
        t1 = (-b - np.sqrt(disc)) / (2*a)
        t2 = (-b + np.sqrt(disc)) / (2*a)
        t0, t3 = sorted((t1, t2))
        s, e = max(0, t0), min(1, t3)
        if e > s:
            intervals.append((s, e))
    # Merge overlapping inside intervals
    intervals.sort(key=lambda x: x[0])
    inside = []
    for a, b in intervals:
        if not inside or a > inside[-1][1]:
            inside.append([a, b])
        else:
            inside[-1][1] = max(inside[-1][1], b)
    # Outside = complement on [0,1]
    outside = []
    prev = 0.0
    for a, b in inside:
        if a > prev:
            outside.append((prev, a))
        prev = b
    if prev < 1.0:
        outside.append((prev, 1.0))
    # Build segments
    segs = []
    # Outside: one segment per interval
    for a, b in outside:
        segs.append((a, b, 'outside'))
    # Inside: overlapping fixed-length segments
    for a, b in inside:
        p0 = P0 + a * D
        p1 = P0 + b * D
        interval_len = np.linalg.norm(p1 - p0)
        if interval_len <= 0:
            continue
        count = ceil(interval_len / fine_len)
        overlap = (count * fine_len - interval_len) / (count - 1) if count > 1 else 0
        dir_unit = (p1 - p0) / interval_len
        start_pt = p0
        for i in range(count):
            end_pt = start_pt + dir_unit * fine_len
            if i == count - 1:
                end_pt = p1
            # compute t0, t1 for table and drawing
            t0 = np.linalg.norm(start_pt - P0) / L
            t1 = np.linalg.norm(end_pt   - P0) / L
            segs.append((t0, t1, 'inside'))
            start_pt = end_pt - dir_unit * overlap
    # Clamp endpoints and convert to points
    segs.sort(key=lambda s: s[0])
    if segs:
        t0, t1, r = segs[0]
        segs[0] = (0.0, t1, r)
        t0, t1, r = segs[-1]
        segs[-1] = (t0, 1.0, r)
    # Convert t-space to points
    out = []
    for t0, t1, reg in segs:
        p0 = P0 + t0 * D
        p1 = P0 + t1 * D
        out.append((p0, p1, reg))
    return out

# ---------- Pygame Visualization ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
sel_idx = None
Drag = None

while True:
    # Compute segments
    segments = split_line_2d(LINE_START, LINE_END, circle_centers, CIRCLE_RADIUS, FINE_LEN)
    # Print once after drag
    if print_now:
        print_table(segments)
        print_now = False
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos; sel_idx = None
            # marker priority
            for idx, (p0, p1, reg) in enumerate(segments):
                d = p1 - p0; n = np.linalg.norm(d)
                perp = np.array([-d[1], d[0]]) / n if n>0 else np.zeros(2)
                sm = p0 + perp * START_MARKER_OFFSET
                em = p1 + perp * END_MARKER_OFFSET
                if np.linalg.norm(np.array([mx,my]) - sm) <= MARKER_RADIUS or \
                   np.linalg.norm(np.array([mx,my]) - em) <= MARKER_RADIUS:
                    sel_idx = idx; break
            if sel_idx is None:
                Drag = None
                for i, C in enumerate(circle_centers):
                    if np.linalg.norm(np.array([mx,my]) - C) <= CIRCLE_RADIUS:
                        Drag = i; break
        elif ev.type == pygame.MOUSEBUTTONUP:
            if Drag is not None:
                print_now = True
            Drag = None
        elif ev.type == pygame.MOUSEMOTION and Drag is not None:
            circle_centers[Drag] = np.array(ev.pos)
    # Draw
    screen.fill((255,255,255))
    for C in circle_centers:
        pygame.draw.circle(screen, (150,150,150), C.astype(int), int(CIRCLE_RADIUS), 2)
    pygame.draw.line(screen, (0,0,0), LINE_START, LINE_END, 2)
    for idx, (p0, p1, reg) in enumerate(segments):
        col = HIGHLIGHT_COLOR if idx==sel_idx else ((200,0,0) if reg=='inside' else (0,0,200))
        width = 5 if idx==sel_idx else 3
        pygame.draw.line(screen, col, p0, p1, width)
        d = p1 - p0; n = np.linalg.norm(d)
        perp = np.array([-d[1], d[0]])/n if n>0 else np.zeros(2)
        sm = p0 + perp*START_MARKER_OFFSET
        em = p1 + perp*END_MARKER_OFFSET
        pygame.draw.circle(screen, START_MARKER_COLOR, sm.astype(int), MARKER_RADIUS)
        pygame.draw.circle(screen, END_MARKER_COLOR,   em.astype(int),   MARKER_RADIUS)
    pygame.display.flip()
    clock.tick(60)
