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

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(dt)
        self.dt = dt 
        self.plane = p.loadURDF("plane.urdf")
        self.reset()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)
        self.obstacles = []
    
    # half extents is measured dist from center of obstacle, reason for this is bc its cheaper to compute 1/2 distance than the entire thing since its symmetrical anyways
    def create_obstacles(self, positions, half_extents=(0.4, 0.4, 0.4)):
        for (x, y) in positions:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[1, 0.4, 0, 1])
            bid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x, y, half_extents[2]])
            self.obstacles.append(bid)

    def check_collision(self):
        for obstacle in self.obstacles:
            if p.getContactPoints(self.car, obstacle):
                return True
        return False
    
    # lidar stuff
    def raycast(self, num_rays=16, fov=np.pi, max_range=10, z=0.2, forward=0.25):
        pos, yaw, _ = self.observe() # current pos

        # raycasts come slightly from in front of the robot
        origin_x = pos[0] + forward * np.cos(yaw)
        origin_y = pos[1] + forward * np.sin(yaw)
        origin = [origin_x, origin_y, z]

        # evenly spaced angles for lidar scans fov is 180 deg
        angles = yaw + np.linspace(-fov / 2, fov / 2, num_rays)

        # from = positions where ray starts, tos = where ray ends
        froms = [origin] * num_rays
        tos = [[origin_x + max_range * np.cos(angle), origin_y, + max_range * np.sin(angle), z] for angle in angles]
        results = p.rayTestBatch(froms, tos) # returns collision info
        return np.array([result[2] for result in results], dtype=np.float32) # hit fraction, 1 = clear
    

    def reset(self, pose=(0,0,0)):
        self.clear_obstacles()
        if hasattr(self, "car"):
            p.removeBody(self.car)
        self.car = p.loadURDF("racecar/racecar.urdf", [pose[0], pose[1], 0.1], p.getQuaternionFromEuler([0, 0, pose[2]]))
        return self.observe()
    
    def apply(self, steering, throttle, max_force = 20, max_speed = 40):
        for s in self.STEER:
            p.setJointMotorControl2(self.car, s, p.POSITION_CONTROL, targetPosition=steering)
        for w in self.WHEELS:
            p.setJointMotorControl2(self.car, w, p.VELOCITY_CONTROL, targetVelocity = throttle * max_speed, force = max_force)

        p.stepSimulation()

    def observe(self):
        pos, orn = p.getBasePositionAndOrientation(self.car)
        yaw = p.getEulerFromQuaternion(orn)[2]
        vel, _ = p.getBaseVelocity(self.car)
        speed = np.hypot(vel[0], vel[1])
        return np.array([pos[0], pos[1]]), yaw, speed

if __name__ == "__main__":
    print_joints()
