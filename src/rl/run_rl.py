import os, sys, time
import numpy as np
import pybullet as p
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from stable_baselines3 import SAC
from car_env import CarSpeedEnv

TARGET_SPEED=2

def main():
    env = CarSpeedEnv(gui=True, target_speed_range=(TARGET_SPEED, TARGET_SPEED))
    model = SAC.load("sac_throttle") # model from train_rl

    t = env.track
    for i in range(len(t)):
        a, b = t[i], t[(i+1) % len(t)] 
        p.addUserDebugLine([a[0], a[1], 0.005], [b[0], b[1], 0.05], [1,0,0], 2)

    obs, _ = env.reset()
    try:
        while True:
            action,_ = model.predict(obs, deterministic=True)
            obs, reward, term ,trunc, info = env.step(action)
            if term or trunc:
                obs, _ = env.reset()

            pos, *_ = env.sim.observe()
            cam = p.getDebugVisualizerCamera()

            p.resetDebugVisualizerCamera(cam[10], cam[8], cam[9], [pos[0, pos[1], 0]])
            time.sleep(env.sim.dt)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

