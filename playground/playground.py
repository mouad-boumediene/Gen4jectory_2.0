"""
Interactive 3D OBB Collision Demo using VPython

- Arrow keys: move blue box in X/Y
- PageUp/PageDown: move in Z
- a/d: yaw left/right
- w/s: pitch up/down
- q/e: roll (spin about box's length axis)

Boxes turn red when colliding, green/blue otherwise.
The faded boxes show the inflated (margin) extents at low opacity.
Requires: `pip install vpython numpy`
"""
from vpython import box, vector, rate, scene, color
import numpy as np

# -------------------------------------------
# Safety margin (in world units) to inflate each box
# -------------------------------------------
margin = 0.5

# -------------------------------------------
# Track pressed keys via event binding
# -------------------------------------------
pressed_keys = set()
def keydown(evt): pressed_keys.add(evt.key)
def keyup(evt):   pressed_keys.discard(evt.key)
scene.bind('keydown', keydown)
scene.bind('keyup', keyup)

# -------------------------------------------
# Helper: rotation matrix (Rodrigues' formula)
# -------------------------------------------
def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[    0, -axis[2],  axis[1]],
                  [ axis[2],     0, -axis[0]],
                  [-axis[1],  axis[0],     0]])
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

# -------------------------------------------
# OBB class with SAT collision and VPython viz
# -------------------------------------------
class OBB:
    def __init__(self, pos, axes, half_sizes, box_color):
        self.pos        = np.array(pos, dtype=float)
        self.axes       = axes.astype(float)
        self.half_sizes = np.array(half_sizes, dtype=float)
        self.vp_color   = box_color
        self._create_boxes()

    def _create_boxes(self):
        # the “real” box
        L, W, H = 2*self.half_sizes
        self.vp = box(pos=vector(*self.pos),
                      axis=vector(*self.axes[0])*L,
                      up=  vector(*self.axes[2])*H,
                      length=L, width=W, height=H,
                      color=self.vp_color,
                      opacity=1.0)
        # the “inflated” box at low opacity
        Li, Wi, Hi = 2*(self.half_sizes+margin)
        self.vpinfl = box(pos=self.vp.pos,
                          axis=vector(*self.axes[0])*Li,
                          up=  vector(*self.axes[2])*Hi,
                          length=Li, width=Wi, height=Hi,
                          color=self.vp_color,
                          opacity=0.2)

    def update_visual(self):
        # update real box
        L, W, H = 2*self.half_sizes
        self.vp.pos   = vector(*self.pos)
        self.vp.axis  = vector(*self.axes[0])*L
        self.vp.up    = vector(*self.axes[2])*H
        self.vp.width = 2*self.half_sizes[1]
        self.vp.color = self.vp_color

        # update inflated box
        Li, Wi, Hi = 2*(self.half_sizes+margin)
        self.vpinfl.pos   = self.vp.pos
        self.vpinfl.axis  = vector(*self.axes[0])*Li
        self.vpinfl.up    = vector(*self.axes[2])*Hi
        self.vpinfl.width = 2*(self.half_sizes[1]+margin)
        self.vpinfl.color = self.vp_color  # match real box’s color

    def collides_with(self, other: "OBB") -> bool:
        A, B = self.axes, other.axes
        a, b = self.half_sizes+margin, other.half_sizes+margin
        R = A @ B.T
        t = A @ (other.pos - self.pos)
        absR = np.abs(R) + 1e-8

        # 1) A’s face-normals
        for i in range(3):
            if abs(t[i]) > a[i] + np.dot(b, absR[i]):
                return False
        # 2) B’s face-normals
        for j in range(3):
            if abs(t @ R[:, j]) > np.dot(a, absR[:, j]) + b[j]:
                return False
        # 3) cross products A_i x B_j
        for i in range(3):
            for j in range(3):
                ra = (a[(i+1)%3] * absR[(i+2)%3, j] +
                      a[(i+2)%3] * absR[(i+1)%3, j])
                rb = (b[(j+1)%3] * absR[i, (j+2)%3] +
                      b[(j+2)%3] * absR[i, (j+1)%3])
                lhs = abs(t[(i+2)%3]*R[(i+1)%3, j] -
                          t[(i+1)%3]*R[(i+2)%3, j])
                if lhs > ra + rb:
                    return False
        return True

# -------------------------------------------
# Set up scene and two boxes
# -------------------------------------------
scene.title = "OBB Collision + Inflated Margin Demo"
scene.caption = (
    "Move the BLUE box, rotate with a/d, w/s, q/e.\n"
    "Original boxes are solid; inflated versions are translucent."
)

axes1 = np.eye(3)
half1 = np.array([1.0, 0.5, 0.3])
box1 = OBB(pos=[0,0,0], axes=axes1, half_sizes=half1, box_color=color.green)

axes2 = np.eye(3)
half2 = np.array([1.0, 0.5, 0.3])
box2 = OBB(pos=[2.5,0,0], axes=axes2, half_sizes=half2, box_color=color.blue)

move_step = 0.1
rot_angle = 5 * np.pi / 180  # 5 degrees

# -------------------------------------------
# Main interaction loop
# -------------------------------------------
while True:
    rate(60)

    # handle movement & rotation of box2
    for key in list(pressed_keys):
        if   key=='left':   box2.pos[0] -= move_step
        elif key=='right':  box2.pos[0] += move_step
        elif key=='up':     box2.pos[1] += move_step
        elif key=='down':   box2.pos[1] -= move_step
        elif key=='pageup':   box2.pos[2] += move_step
        elif key=='pagedown': box2.pos[2] -= move_step
        elif key in ('a','d','w','s','q','e'):
            if key in ('a','d'): axis = box2.axes[2]  # yaw
            if key in ('w','s'): axis = box2.axes[1]  # pitch
            if key in ('q','e'): axis = box2.axes[0]  # roll
            angle = rot_angle * (1 if key in ('d','s','e') else -1)
            R = rotation_matrix(axis, angle)
            box2.axes = (R @ box2.axes.T).T

    # collision check (with margin)
    hit = box1.collides_with(box2)
    box1.vp_color = color.red if hit else color.green
    box2.vp_color = color.red if hit else color.blue

    # update both the solid and inflated visuals
    box1.update_visual()
    box2.update_visual()
