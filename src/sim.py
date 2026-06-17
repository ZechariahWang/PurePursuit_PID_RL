import pybullet as p, pybullet_data
import numpy as np

def print_joints(gui=False):
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    car = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.1])
    for i in range(p.getNumJoints(car)):
        info = p.getJointInfo(car, i)
        print(i, info[1].decode(), info[2])
    p.disconnect()

class CarSim:
    WHEELS = [2, 3 ,5, 7]
    STEER = [4, 6]
    WHEELBASE = 0.325  # meter

    def __init__(self, gui=True, dt = 1/60):
        self.client=p.connect(p.GUI if gui else p.DIRECT)
        self.obstacles = []

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(dt, physicsClientId=self.client)
        self.dt = dt
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.reset()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle, physicsClientId=self.client)
        self.obstacles = []

    # half extents is measured dist from center of obstacle, reason for this is bc its cheaper to compute 1/2 distance than the entire thing since its symmetrical anyways
    def create_obstacles(self, positions, half_extents=(0.4, 0.4, 0.4)):
        for (x, y) in positions:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0.4, 0, 1], physicsClientId=self.client)
            bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x, y, half_extents[2]], physicsClientId=self.client)
            self.obstacles.append(bid)

    def check_collision(self):
        for obstacle in self.obstacles:
            if p.getContactPoints(self.car, obstacle, physicsClientId=self.client):
                return True
        return False
    
    # lidar stuff. forward must clear the chassis (~0.4m) or rays self-hit the car
    def raycast(self, num_rays=8, fov=np.pi, max_range=10, z=0.2, forward=0.4):
        pos, yaw, _ = self.observe() # current pos

        # raycasts come slightly from in front of the robot
        origin_x = pos[0] + forward * np.cos(yaw)
        origin_y = pos[1] + forward * np.sin(yaw)
        origin = [origin_x, origin_y, z]

        # evenly spaced angles for lidar scans fov is 180 deg
        angles = yaw + np.linspace(-fov / 2, fov / 2, num_rays)

        # from = positions where ray starts, tos = where ray ends
        froms = [origin] * num_rays
        tos = [[origin_x + max_range * np.cos(angle), origin_y + max_range * np.sin(angle), z] for angle in angles]
        results = p.rayTestBatch(froms, tos, physicsClientId=self.client) # returns collision info

        # treat rays that hit the car itself as clear (1.0), else report hit fraction, 1 = clear
        return np.array([1.0 if r[0] == self.car else r[2] for r in results], dtype=np.float32)

    # same rays as raycast() but also returns the start/hit points so run_rl can draw them
    def raycast_points(self, num_rays=16, fov=np.pi, max_range=10, z=0.2, forward=0.4):
        pos, yaw, _ = self.observe()
        origin_x = pos[0] + forward * np.cos(yaw)
        origin_y = pos[1] + forward * np.sin(yaw)
        origin = [origin_x, origin_y, z]

        angles = yaw + np.linspace(-fov / 2, fov / 2, num_rays)
        froms = [origin] * num_rays
        tos = [[origin_x + max_range * np.cos(angle), origin_y + max_range * np.sin(angle), z] for angle in angles]
        fracs = np.array([1.0 if r[0] == self.car else r[2] for r in p.rayTestBatch(froms, tos, physicsClientId=self.client)], dtype=np.float32)
        hits = [[origin_x + f * max_range * np.cos(a), origin_y + f * max_range * np.sin(a), z]
                for f, a in zip(fracs, angles)]
        return froms, hits, fracs

    def reset(self, pose=(0,0,0)):
        self.clear_obstacles()
        if hasattr(self, "car"):
            p.removeBody(self.car, physicsClientId=self.client)
        self.car = p.loadURDF("racecar/racecar.urdf", [pose[0], pose[1], 0.1],
                              p.getQuaternionFromEuler([0, 0, pose[2]]), physicsClientId=self.client)
        return self.observe()

    def apply(self, steering, throttle, max_force = 20, max_speed = 40):
        for s in self.STEER:
            p.setJointMotorControl2(self.car, s, p.POSITION_CONTROL, targetPosition=steering, physicsClientId=self.client)
        for w in self.WHEELS:
            p.setJointMotorControl2(self.car, w, p.VELOCITY_CONTROL, targetVelocity = throttle * max_speed, force = max_force, physicsClientId=self.client)

        p.stepSimulation(physicsClientId=self.client)

    def observe(self):
        pos, orn = p.getBasePositionAndOrientation(self.car, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
        vel, _ = p.getBaseVelocity(self.car, physicsClientId=self.client)
        speed = np.hypot(vel[0], vel[1])
        return np.array([pos[0], pos[1]]), yaw, speed

if __name__ == "__main__":
    print_joints()
