"""
Interactive 3D OBB Collision Demo using VPython

- Arrow keys: move blue box in X/Y
- PageUp/PageDown: move in Z
- a/d: yaw left/right
- w/s: pitch up/down
- q/e: roll (spin about box's length axis)

Boxes turn red when colliding, green/blue otherwise.
Requires: `pip install vpython numpy`
"""
from vpython import box, vector, rate, scene, color
import numpy as np

# -------------------------------------------
# Track pressed keys via event binding
# -------------------------------------------
pressed_keys = set()

def keydown(evt):
    pressed_keys.add(evt.key)

def keyup(evt):
    pressed_keys.discard(evt.key)

# Bind keydown and keyup events
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
    def __init__(self,
                 pos: np.ndarray,
                 axes: np.ndarray,
                 half_sizes: np.ndarray,
                 box_color=color.blue):
        # pos: (3,), axes: (3,3) each row is local axis, half_sizes: (3,)
        self.pos = np.array(pos, dtype=float)
        self.axes = axes.astype(float)
        self.half_sizes = np.array(half_sizes, dtype=float)
        self.vp_color = box_color
        self._create_vpython_box()

    def _create_vpython_box(self):
        length = 2 * self.half_sizes[0]
        width  = 2 * self.half_sizes[1]
        height = 2 * self.half_sizes[2]
        self.vp = box(
            pos=vector(*self.pos),
            axis=vector(*self.axes[0]) * length,
            up=vector(*self.axes[2]) * height,
            length=length,
            width=width,
            height=height,
            color=self.vp_color
        )

    def update_visual(self):
        length = 2 * self.half_sizes[0]
        height = 2 * self.half_sizes[2]
        self.vp.pos  = vector(*self.pos)
        self.vp.axis = vector(*self.axes[0]) * length
        self.vp.up   = vector(*self.axes[2]) * height
        self.vp.width = 2 * self.half_sizes[1]
        self.vp.color = self.vp_color

    def collides_with(self, other: "OBB") -> bool:
        A = self.axes
        B = other.axes
        a = self.half_sizes
        b = other.half_sizes
        R = A @ B.T
        t = A @ (other.pos - self.pos)
        absR = np.abs(R) + 1e-8
        for i in range(3):
            if abs(t[i]) > a[i] + np.dot(b, absR[i]):
                return False
        for j in range(3):
            if abs(t @ R[:, j]) > np.dot(a, absR[:, j]) + b[j]:
                return False
        for i in range(3):
            for j in range(3):
                ra = a[(i+1)%3] * absR[(i+2)%3, j] + a[(i+2)%3] * absR[(i+1)%3, j]
                rb = b[(j+1)%3] * absR[i, (j+2)%3] + b[(j+2)%3] * absR[i, (j+1)%3]
                lhs = abs(t[(i+2)%3] * R[(i+1)%3, j] - t[(i+1)%3] * R[(i+2)%3, j])
                if lhs > ra + rb:
                    return False
        return True

# -------------------------------------------
# Set up scene and boxes
# -------------------------------------------
scene.title = "OBB Collision Demo"
scene.caption = (
    "Use arrow keys to move the BLUE box (X/Y), PageUp/PageDown for Z.\n"
    "a/d: yaw   w/s: pitch   q/e: roll  (about its local axes)\n"
    "Boxes turn RED when colliding."
)

axes1 = np.eye(3)
half1 = np.array([1.0, 0.5, 0.3])
box1 = OBB(pos=np.array([0.0, 0.0, 0.0]), axes=axes1, half_sizes=half1, box_color=color.green)

axes2 = np.eye(3)
half2 = np.array([1.0, 0.5, 0.3])
box2 = OBB(pos=np.array([2.5, 0.0, 0.0]), axes=axes2, half_sizes=half2, box_color=color.blue)

move_step = 0.1
rot_angle = 5 * np.pi / 180  # 5 degrees

# Main loop
while True:
    rate(60)
    for key in list(pressed_keys):
        if key == 'left':   box2.pos[0] -= move_step
        elif key == 'right':  box2.pos[0] += move_step
        elif key == 'up':     box2.pos[1] += move_step
        elif key == 'down':   box2.pos[1] -= move_step
        elif key == 'pageup':   box2.pos[2] += move_step
        elif key == 'pagedown': box2.pos[2] -= move_step
        elif key in ('a','d','w','s','q','e'):
            if key in ('a','d'): axis = box2.axes[2]  # yaw (local Z)
            if key in ('w','s'): axis = box2.axes[1]  # pitch (local Y)
            if key in ('q','e'): axis = box2.axes[0]  # roll (local X)
            angle = rot_angle * (1 if key in ('d','s','e') else -1)
            R = rotation_matrix(axis, angle)
            box2.axes = (R @ box2.axes.T).T
    
    # check collision and update colors
    collision = box1.collides_with(box2)
    box1.vp_color = color.red if collision else color.green
    box2.vp_color = color.red if collision else color.blue
    box1.update_visual()
    box2.update_visual()
