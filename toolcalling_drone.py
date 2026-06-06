#!/usr/bin/env python3
"""
toolcalling_drone.py

Tool-calling agent for driving a simulated drone in 3D, visualized with simple point clouds.

Designed for resource-sparse Orin (Jetson) + small Ollama model (e.g. lfm2.5-thinking or gemma3:1b).

Goal: Let the LLM use natural language + tool calls to control a drone:
- Move and turn in 3D space
- "Scan" the environment (generates/updates a point cloud "sensor" view and map)
- Reason about position and obstacles from tool feedback

Visualization uses lightweight PyOpenGL + GLUT point clouds (no extra heavy deps):
- Static environment points (floor + obstacles) in gray
- Accumulated map / scan points (updated live by the model) in green
- Drone represented as a small moving set of red points + heading
- Orange flight path trail

User can rotate/zoom the 3D view with mouse while the model "flies".

This is the evolution of your file CRUD + 3D tool-calling experiments, now aimed at drone control.

Run (in your Orin venv that has ollama):
    source /path/to/venv/bin/activate
    # No extra pip install needed (uses PyOpenGL + GLUT you already had)
    python toolcalling_drone.py

Controls in 3D window:
- Left mouse: rotate
- Right mouse or wheel: zoom/pan
- The model drives via tools; you observe and can talk to it.

Example prompts for the model:
"fly forward 3 meters then scan"
"turn left 90 degrees, move forward until you see a wall on the right"
"explore the area by scanning and mapping as you go"

The model gets compact feedback like current pose and scan summaries after each tool.
"""

import threading
import queue
import time
import json
import numpy as np
import math
from dataclasses import dataclass

# GLUT / OpenGL for lightweight 3D point cloud rendering (no heavy deps)
# This was confirmed working on your system earlier (PyOpenGL + freeglut present)
try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    HAS_GLUT = True
except ImportError:
    HAS_GLUT = False

try:
    import ollama
except ImportError:
    ollama = None

# =============================================================================
# Drone Simulation + Point Cloud "World"
# =============================================================================

@dataclass
class DroneState:
    x: float = 0.0
    y: float = 1.0   # height
    z: float = 0.0
    yaw: float = 0.0  # degrees, 0 = +z forward

class DroneSimulator:
    def __init__(self):
        self.state = DroneState()
        self.lock = threading.Lock()
        self.path = []  # list of (x,y,z) for trail
        self.map_points = []  # accumulated "known" points from scans (Nx3)
        self.last_scan_points = []  # most recent scan
        self.last_action = "Drone initialized at origin"

        # Simple static environment: floor + a few "obstacles/walls" as point clouds
        self.env_points = self._generate_environment()

    def _generate_environment(self):
        """Generate a simple point cloud 'world' the drone can fly in and scan."""
        points = []
        # Floor grid - coarser for performance on Orin (GL immediate mode is slow for 10k+ points)
        for x in np.arange(-15, 16, 1.5):
            for z in np.arange(-15, 16, 1.5):
                points.append([x, 0.0, z])

        # Some wall/obstacle clusters - sparser
        # Wall on left
        for y in np.linspace(0, 4, 5):
            for z in np.arange(-8, 8, 1.0):
                points.append([-8.0, y, z])
        # Wall on right with gap
        for y in np.linspace(0, 4, 5):
            for z in np.arange(-8, -1, 1.0):
                points.append([8.0, y, z])
            for z in np.arange(1, 8, 1.0):
                points.append([8.0, y, z])
        # Far wall
        for y in np.linspace(0, 3, 4):
            for x in np.arange(-6, 6, 1.2):
                points.append([x, y, 10.0])

        # A couple of pillar clusters - fewer points
        for cx, cz in [(-4, -4), (3, 5)]:
            for dx in np.arange(-1.2, 1.3, 0.8):
                for dz in np.arange(-1.2, 1.3, 0.8):
                    for y in np.linspace(0, 2.5, 4):
                        points.append([cx + dx, y, cz + dz])

        return np.array(points)

    def get_pose(self):
        with self.lock:
            return {
                "x": round(self.state.x, 2),
                "y": round(self.state.y, 2),
                "z": round(self.state.z, 2),
                "yaw": round(self.state.yaw, 1),
                "last_action": self.last_action
            }

    def move_forward(self, distance: float = 1.0):
        with self.lock:
            rad = np.deg2rad(self.state.yaw)
            self.state.x += distance * np.sin(rad)   # note: yaw 0 faces +Z, x is right
            self.state.z += distance * np.cos(rad)
            self.state.y = max(0.2, self.state.y)  # don't go underground
            self.path.append((self.state.x, self.state.y, self.state.z))
            self.last_action = f"Moved forward {distance}m"
            # simple "crash" if too low or out of bounds for demo
            if abs(self.state.x) > 14 or abs(self.state.z) > 14:
                self.last_action += " (near boundary)"
            return f"OK: new pose x={self.state.x:.2f} y={self.state.y:.2f} z={self.state.z:.2f} yaw={self.state.yaw:.1f}"

    def turn(self, degrees: float = 30.0):
        with self.lock:
            self.state.yaw = (self.state.yaw + degrees) % 360
            self.last_action = f"Turned {degrees} degrees"
            return f"OK: yaw now {self.state.yaw:.1f}°"

    def change_height(self, delta: float = 1.0):
        with self.lock:
            self.state.y = max(0.2, min(8.0, self.state.y + delta))
            self.last_action = f"Changed height by {delta}m"
            return f"OK: height now {self.state.y:.2f}m"

    def scan(self, range_m: float = 4.0, density: int = 80):
        """
        Simulate a simple LiDAR / depth scan in front of the drone.
        Adds points to the map and returns a summary + the new points.
        """
        with self.lock:
            rad = np.deg2rad(self.state.yaw)
            new_points = []
            # Fan of rays in front (simple approximation, no real raycasting)
            for angle_offset in np.linspace(-45, 45, 9):
                for dist in np.linspace(0.5, range_m, 6):
                    a = rad + np.deg2rad(angle_offset)
                    px = self.state.x + dist * np.sin(a)
                    pz = self.state.z + dist * np.cos(a)
                    # height variation + some noise to look like real scan
                    py = self.state.y + np.random.uniform(-0.3, 0.8)
                    new_points.append([px, py, pz])

            new_points = np.array(new_points)

            # Add to accumulated map (downsample a bit mentally by keeping every N)
            if len(self.map_points) == 0:
                self.map_points = new_points.tolist()
            else:
                self.map_points.extend(new_points.tolist())
                # crude limit to keep it light on Orin
                if len(self.map_points) > 2500:
                    self.map_points = self.map_points[-2000:]

            self.last_scan_points = new_points.tolist()
            self.last_action = f"Scanned {len(new_points)} points (range {range_m}m)"

            # Return compact info the LLM can use
            summary = f"Scanned {len(new_points)} points. Closest point dist ~{np.min(np.linalg.norm(new_points - [self.state.x, self.state.y, self.state.z], axis=1)):.2f}m"
            return f"OK: {summary}. Map now has ~{len(self.map_points)} points total."

    def get_status(self):
        with self.lock:
            pose = self.get_pose()
            scan_info = f"Last scan: {len(self.last_scan_points)} points" if self.last_scan_points else "No scan yet"
            map_info = f"Map size: {len(self.map_points)} points"
            return f"Pose: x={pose['x']} y={pose['y']} z={pose['z']} yaw={pose['yaw']} | {scan_info} | {map_info} | Last: {pose['last_action']}"

    def reset(self):
        with self.lock:
            self.state = DroneState()
            self.path = []
            self.map_points = []
            self.last_scan_points = []
            self.last_action = "Reset to origin"
            return "OK: drone reset"

    def get_drone_points(self):
        """Small point cloud representing the drone + heading for visualization."""
        with self.lock:
            s = self.state
            rad = np.deg2rad(s.yaw)
            # Body center
            center = np.array([s.x, s.y, s.z])
            # Simple "drone" as 5 points: center + 4 arms (cross + forward indicator)
            arm_len = 0.6
            points = [
                center,
                center + [arm_len, 0, 0],
                center + [-arm_len, 0, 0],
                center + [0, 0, arm_len],      # forward
                center + [0, arm_len*0.5, 0],  # up a bit
            ]
            # Heading indicator (further forward)
            points.append(center + [0, 0.2, arm_len * 1.8])
            return np.array(points)

    def get_path_lines(self):
        with self.lock:
            if len(self.path) < 2:
                return None
            return np.array(self.path)


# =============================================================================
# Lightweight GLUT Visualization (point clouds via GL_POINTS)
# Re-uses the proven PyOpenGL+GLUT that was already working on your system.
# Renders simple point clouds for env, map/scans, drone, and path.
# No extra deps beyond PyOpenGL+GLUT (already present).
# =============================================================================

class DroneVisualizer:
    def __init__(self, sim: DroneSimulator):
        if not HAS_GLUT:
            raise RuntimeError(
                "PyOpenGL + GLUT not available. "
                "They were present on your system (confirmed earlier). "
                "Install with: pip install PyOpenGL PyOpenGL_accelerate"
            )
        self.sim = sim
        self.width = 900
        self.height = 700

        # Camera (orbit style)
        self.cam_yaw = 45.0
        self.cam_pitch = 25.0
        self.cam_distance = 15.0
        self.cam_target = [0.0, 2.0, 0.0]

        self.mouse_down = False
        self.last_mouse = (0, 0)
        self.running = True

    def init_gl(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(b"Drone Point Cloud Sandbox (LLM Controlled) - GLUT")

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glPointSize(2.5)  # good size for simple points on Orin

    def reshape(self, w, h):
        self.width, self.height = w, h
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / float(h) if h > 0 else 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera
        eye_x = self.cam_target[0] + self.cam_distance * math.cos(math.radians(self.cam_yaw)) * math.cos(math.radians(self.cam_pitch))
        eye_y = self.cam_target[1] + self.cam_distance * math.sin(math.radians(self.cam_pitch))
        eye_z = self.cam_target[2] + self.cam_distance * math.sin(math.radians(self.cam_yaw)) * math.cos(math.radians(self.cam_pitch))

        gluLookAt(eye_x, eye_y, eye_z,
                  self.cam_target[0], self.cam_target[1], self.cam_target[2],
                  0, 1, 0)

        # Floor grid
        glColor3f(0.25, 0.25, 0.28)
        glBegin(GL_LINES)
        for i in range(-12, 13, 2):
            glVertex3f(i, 0, -12)
            glVertex3f(i, 0, 12)
            glVertex3f(-12, 0, i)
            glVertex3f(12, 0, i)
        glEnd()

        # Axes
        glColor3f(1, 0, 0); glBegin(GL_LINES); glVertex3f(0,0,0); glVertex3f(2,0,0); glEnd()
        glColor3f(0, 1, 0); glBegin(GL_LINES); glVertex3f(0,0,0); glVertex3f(0,2,0); glEnd()
        glColor3f(0, 0, 1); glBegin(GL_LINES); glVertex3f(0,0,0); glVertex3f(0,0,2); glEnd()

        # Fetch fresh data (under lock, copy to avoid long lock)
        with self.sim.lock:
            env_pts = np.array(self.sim.env_points) if len(self.sim.env_points) > 0 else np.zeros((0, 3))
            map_pts = np.array(self.sim.map_points) if len(self.sim.map_points) > 0 else np.zeros((0, 3))
            drone_pts = self.sim.get_drone_points()
            path_pts = self.sim.get_path_lines()

        # Environment points (gray) - subsample for perf (immediate mode GL slow on Orin for dense clouds)
        if len(env_pts) > 0:
            glColor3f(0.55, 0.55, 0.6)
            glBegin(GL_POINTS)
            for p in env_pts[::2]:  # every other point
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

        # Mapped/scanned points (green) - subsample
        if len(map_pts) > 0:
            glColor3f(0.2, 0.85, 0.3)
            glBegin(GL_POINTS)
            for p in map_pts[::2]:
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

        # Drone (red, slightly larger points)
        glPointSize(5.0)
        glColor3f(1.0, 0.15, 0.15)
        glBegin(GL_POINTS)
        for p in drone_pts:
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()
        glPointSize(2.5)  # reset

        # Path (orange line strip)
        if path_pts is not None and len(path_pts) >= 2:
            glColor3f(1.0, 0.55, 0.0)
            glBegin(GL_LINE_STRIP)
            for p in path_pts:
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

        glutSwapBuffers()

    def mouse(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON:
            self.mouse_down = (state == GLUT_DOWN)
            self.last_mouse = (x, y)
        elif button == 3:  # wheel up
            self.cam_distance = max(3.0, self.cam_distance - 0.6)
        elif button == 4:  # wheel down
            self.cam_distance = min(50.0, self.cam_distance + 0.6)
        glutPostRedisplay()

    def motion(self, x, y):
        if self.mouse_down:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            self.cam_yaw += dx * 0.4
            self.cam_pitch = max(-85, min(85, self.cam_pitch - dy * 0.4))
            self.last_mouse = (x, y)
            glutPostRedisplay()

    def keyboard(self, key, x, y):
        key = key.decode("utf-8").lower() if isinstance(key, bytes) else key.lower()
        if key == '\x1b':  # ESC
            self.running = False
            glutLeaveMainLoop()
        elif key == 'r':
            self.cam_yaw = 45.0
            self.cam_pitch = 25.0
            self.cam_distance = 15.0
            glutPostRedisplay()
        elif key == 'c':
            self.sim.reset()
            glutPostRedisplay()

    def idle(self):
        glutPostRedisplay()
        time.sleep(0.016)

    def run(self):
        self.init_gl()

        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        glutKeyboardFunc(self.keyboard)
        glutIdleFunc(self.idle)

        print("\n[GLUT] Drone Point Cloud Viewer ready (lightweight, using your existing PyOpenGL+GLUT).")
        print("Left-drag: orbit camera | Mouse wheel: zoom | 'r': reset camera | 'c': reset drone | ESC: quit")
        print("Use the terminal to give the model commands to drive the drone.\n")

        glutMainLoop()


# =============================================================================
# Tool definitions for the LLM (drone control)
# =============================================================================

def make_drone_tools(sim: DroneSimulator):
    return [
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": "Get the current drone position (x,y,z), yaw, and summary of what has been scanned/mapped. Call this often to know where you are.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "move_forward",
                "description": "Fly straight forward in the current yaw direction by the given distance in meters. Use small values (0.5-2.0) for precision.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "distance": {"type": "number", "description": "meters to move forward"}
                    },
                    "required": ["distance"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "turn",
                "description": "Turn the drone in place by degrees. Positive = left (counter-clockwise), negative = right.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "degrees": {"type": "number"}
                    },
                    "required": ["degrees"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "change_height",
                "description": "Change altitude. Positive = up, negative = down. Keep between 0.5 and 6m for safety.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "delta": {"type": "number", "description": "meters to change height (+ up, - down)"}
                    },
                    "required": ["delta"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "scan",
                "description": "Perform a sensor scan in front of the drone. This updates the 3D point cloud map the user sees and gives you distance info. Use frequently while exploring.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "range_m": {"type": "number", "description": "how far to scan (default 4.0)"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reset_drone",
                "description": "Reset the drone to the starting position and clear the map. Use when stuck or for a new mission.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
    ]


# =============================================================================
# Agent (same style as your previous tool-calling scripts)
# =============================================================================

SYSTEM_PROMPT = """You are an autonomous drone pilot agent running on a small local LLM.

Your job is to safely fly the drone around a 3D environment using only tool calls.

MANDATORY WORKFLOW:
1. Always call get_status first to know your current position, yaw, and what you have already mapped.
2. Use scan often to "see" the world as point clouds (the user watches the 3D window).
3. Move in small, deliberate steps. Turn to face interesting directions before moving.
4. Build a mental map from the scan results and status.
5. Avoid flying into walls (if scan shows points very close in your flight direction, turn or climb).
6. When you have explored or reached a goal, describe what you found.

You control a real (simulated) drone the user can watch in 3D. Be precise with distances and angles.
The 3D view shows:
- Gray points = static environment
- Green points = what you have scanned/mapped so far
- Red points = current drone location + heading
- Orange line = your flight path

Speak concisely. After actions, the tools will give you updated status.
"""

def run_drone_agent(sim: DroneSimulator, model: str = "lfm2.5-thinking"):
    tools = make_drone_tools(sim)

    # Bind the actual methods
    impls = {
        "get_status": sim.get_status,
        "move_forward": sim.move_forward,
        "turn": sim.turn,
        "change_height": sim.change_height,
        "scan": sim.scan,
        "reset_drone": sim.reset,
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Drone Pilot Agent ready (small LLM + point cloud viz).")
    print("Give the drone a mission in natural language, e.g.:")
    print("  'explore the area safely by scanning and moving around the obstacles'")
    print("  'fly to the far wall on the right side, scanning as you go'")
    print("Type 'exit' or 'reset' anytime.\n")

    while True:
        try:
            user = input("Mission / command: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user.lower() in ("exit", "quit", "q"):
            break
        if user.lower() == "reset":
            print(sim.reset_drone())
            continue
        if not user:
            continue

        messages.append({"role": "user", "content": user})

        # Give the model fresh state so it doesn't lose track
        status = sim.get_status()
        messages.append({"role": "system", "content": f"[CURRENT DRONE STATUS] {status}"})

        try:
            if ollama is None:
                print("ollama package not available. Install in your venv.")
                break
            resp = ollama.chat(model=model, messages=messages, tools=tools)
        except Exception as e:
            print(f"[ollama error] {e}")
            continue

        msg = resp.get("message", {})
        messages.append(msg)

        content = (msg.get("content") or "").strip()
        if content:
            print("Pilot:", content)

        for tc in msg.get("tool_calls", []):
            name = tc["function"]["name"]
            args = tc["function"].get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}

            impl = impls.get(name)
            if impl:
                try:
                    result = impl(**args) if args else impl()
                except Exception as e:
                    result = f"ERROR: {e}"
            else:
                result = f"Unknown tool: {name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": str(result)
            })
            print(f"[tool:{name}] {result}")

        # Trim very old messages to keep context small for tiny models
        if len(messages) > 18:
            messages = messages[:4] + messages[-12:]  # keep system + recent


# =============================================================================
# Reusable DroneAgent class (for testing and embedding, parallel to FileAgent)
# =============================================================================

class DroneAgent:
    """Reusable agent for drone control via tool calls.
    Use this from the test harness:
        agent = td.DroneAgent(model="lfm2.5-thinking")
        result = agent.send("fly forward 2 and scan")
        # then inspect agent.sim or result
    """

    def __init__(self, model: str = "lfm2.5-thinking"):
        self.model = model
        self.sim = DroneSimulator()
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tools = make_drone_tools(self.sim)
        self.impls = {
            "get_status": self.sim.get_status,
            "move_forward": self.sim.move_forward,
            "turn": self.sim.turn,
            "change_height": self.sim.change_height,
            "scan": self.sim.scan,
            "reset_drone": self.sim.reset,
        }
        self.last_result: dict = {}

    def reset(self):
        """Reset sim and conversation for a fresh test."""
        self.sim.reset()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, user_message: str, max_tool_rounds: int = 8) -> dict:
        """Send a natural language command/mission. Runs tool loop until model stops.
        Returns dict with 'assistant', 'tools_used', 'final_pose', 'map_size', etc.
        """
        self.messages.append({"role": "user", "content": user_message})

        # Always inject fresh status so small model knows the world
        status = self.sim.get_status()
        self.messages.append({"role": "system", "content": f"[CURRENT DRONE STATUS] {status}"})

        tools_used: list[dict] = []
        assistant_text = ""

        for _ in range(max_tool_rounds):
            if ollama is None:
                raise RuntimeError("ollama package not installed")
            try:
                resp = ollama.chat(model=self.model, messages=self.messages, tools=self.tools)
            except Exception as e:
                return {"assistant": "", "tools_used": [], "error": str(e), "final_pose": self.sim.get_pose()}

            msg = resp.get("message", {}) or {}
            self.messages.append(msg)

            if msg.get("content"):
                assistant_text = msg["content"].strip()

            for tc in msg.get("tool_calls", []):
                name = tc["function"]["name"]
                args = tc["function"].get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                impl = self.impls.get(name)
                if impl:
                    try:
                        result = impl(**args) if args else impl()
                    except Exception as e:
                        result = f"ERROR in {name}: {e}"
                else:
                    result = f"Unknown tool {name}"

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": str(result)
                })

                tools_used.append({
                    "name": name,
                    "args": args,
                    "result": result
                })

            if not msg.get("tool_calls"):
                break

        # Trim history to keep small for tiny models
        if len(self.messages) > 18:
            self.messages = self.messages[:4] + self.messages[-12:]

        pose = self.sim.get_pose()
        map_size = len(self.sim.map_points)

        self.last_result = {
            "assistant": assistant_text,
            "tools_used": tools_used,
            "final_pose": pose,
            "map_size": map_size,
            "last_action": self.sim.last_action,
        }
        return self.last_result

    def get_sim(self) -> DroneSimulator:
        return self.sim


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== LLM Drone Point Cloud Controller ===")
    print("Using lightweight GLUT point clouds for visualization (Orin-friendly).")

    sim = DroneSimulator()

    # Seed a small initial scan so the map isn't empty
    sim.scan(range_m=2.5)

    vis = DroneVisualizer(sim)

    # Run the LLM agent in background thread (it will call tools that mutate sim)
    agent_thread = threading.Thread(
        target=run_drone_agent, args=(sim,), daemon=True
    )
    agent_thread.start()

    # Main thread owns the GLUT window and render loop
    vis.run()

    print("Viewer closed. Agent thread will exit shortly.")


if __name__ == "__main__":
    main()
