import os
import sys
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from track import make_track, nearest_index, lookahead_point
from pure_pursuit import steering_angle
from pid import PID
from sim import CarSim

WAYPOINTS = [(0, 0), (30, 5), (40, 25), (20, 40), (-10, 30), (-20, 5)]
TARGET_SPEED = 2.0 
LD = 2.0             
MAX_STEER = 0.6
FOLLOW_CAM = True # cam on car

def draw_track(track):
    n = len(track)
    for k in range(n):
        a, b = track[k], track[(k + 1) % n]
        p.addUserDebugLine([a[0], a[1], 0.05], [b[0], b[1], 0.05], lineColorRGB=[1, 0, 0], lineWidth=2)
def main():
    track = make_track(WAYPOINTS, samples_per_seg=200)
    start = track[0]
    heading = np.arctan2(track[1, 1] - track[0, 1], track[1, 0] - track[0, 0])

    sim = CarSim(gui=True)
    sim.reset(pose=(start[0], start[1], heading))
    draw_track(track)
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-35, cameraTargetPosition=[start[0], start[1], 0])

    pid = PID(kp=2.0, ki=0.5, kd=0.0)

    try:
        while True:
            pos, yaw, speed = sim.observe()
            idx = nearest_index(track, pos)
            look_pt, _ = lookahead_point(track, pos, LD, idx)

            steer = steering_angle(pos, yaw, look_pt, sim.WHEELBASE, LD)
            steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))
            throttle = pid.step(TARGET_SPEED, speed, sim.dt)

            sim.apply(steer, throttle)

            if FOLLOW_CAM: # ctrl + drag
                cam = p.getDebugVisualizerCamera()
                yaw, pitch, dist = cam[8], cam[9], cam[10]
                p.resetDebugVisualizerCamera(dist, yaw, pitch, [pos[0], pos[1], 0])
            time.sleep(sim.dt) 

    except KeyboardInterrupt:
        print("\nstopped")

if __name__ == "__main__":
    main()
