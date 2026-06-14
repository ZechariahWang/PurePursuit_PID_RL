import os, sys, time
import numpy as np
import pybullet as p
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from car_env import CarSpeedEnv as ThrottleEnv
from new_car_env import CarSpeedEnv as NavEnv, NUM_RAYS, LIDAR_RANGE

USE_NAV_MODEL   = True 
TARGET_SPEED    = 2 

# to open logs, tensorboard --logdir tb

def main():
    if USE_NAV_MODEL:
        env = NavEnv(gui=True)
        model = SAC.load("sac_nav")
    else:
        env = ThrottleEnv(gui=True, target_speed_range=(TARGET_SPEED, TARGET_SPEED))
        model = SAC.load("sac_throttle")

    t = env.track
    for i in range(len(t)):
        a, b = t[i], t[(i+1) % len(t)]
        p.addUserDebugLine([a[0], a[1], 0.05], [b[0], b[1], 0.05], [1,0,0], 2) # line start pos, line end pos, color (rgb normalized), lind size

    ray_ids = [-1] * NUM_RAYS  # reused each frame so lidar lines don't pile up
    obs, _ = env.reset()
    try:
        while True:
            action,_ = model.predict(obs, deterministic=True)
            obs, reward, term ,trunc, info = env.step(action)
            if term or trunc:
                obs, _ = env.reset()

            if USE_NAV_MODEL:
                froms, hits, fracs = env.sim.raycast_points(num_rays=NUM_RAYS, max_range=LIDAR_RANGE)
                for k in range(NUM_RAYS):
                    color = [1, 0, 0] if fracs[k] < 1.0 else [0, 1, 0]  # red = hit, green = clear
                    ray_ids[k] = p.addUserDebugLine(froms[k], hits[k], color, 1, replaceItemUniqueId=ray_ids[k])

            pos, *_ = env.sim.observe()
            cam = p.getDebugVisualizerCamera()

            p.resetDebugVisualizerCamera(cam[10], cam[8], cam[9], [pos[0], pos[1], 0]) # distance, yaw angle, pitch angle, target pos
            time.sleep(env.sim.dt)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
