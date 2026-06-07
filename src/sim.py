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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(dt)
        self.dt = dt 
        self.plane = p.loadURDF("plane.urdf")
        self.reset()

    def reset(self, pose=(0,0,0)):
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
