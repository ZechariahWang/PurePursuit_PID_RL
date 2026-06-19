import os, sys, time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC

from rl.environments.goal_nav_env import GoalNavEnv, NUM_RAYS, LIDAR_RANGE
from rl.environments.path_nav_env import CarSpeedEnv as NavEnv
from rl.environments.car_env import CarSpeedEnv as ThrottleEnv

MODE               = "goal" # goal, nav, throttle
TARGET_SPEED       = 2.0    # only used by throttle

# tensorboard --logdir tb

def draw_goal(env, ids):
    gx, gy = env.goal
    pole = p.addUserDebugLine([gx, gy, 0], [gx, gy, 1.5], [0, 0, 1], 3,
                              replaceItemUniqueId=ids[0] if ids else -1)
    txt = p.addUserDebugText("GOAL", [gx, gy, 1.7], [0, 0, 1], 1.2,
                             replaceItemUniqueId=ids[1] if ids else -1)
    return [pole, txt]

def draw_track(env):
    t = env.track
    for i in range(len(t)):
        a, b = t[i], t[(i + 1) % len(t)]
        p.addUserDebugLine([a[0], a[1], 0.05], [b[0], b[1], 0.05], [1, 0, 0], 2)

def main():
    if MODE == "goal":
        env, model, show_lidar = GoalNavEnv(gui=True), SAC.load("sac_goal_nav"), True
    elif MODE == "nav":
        env, model, show_lidar = NavEnv(gui=True), SAC.load("sac_nav"), True
    elif MODE == "throttle":
        env = ThrottleEnv(gui=True, target_speed_range=(TARGET_SPEED, TARGET_SPEED))
        model, show_lidar = SAC.load("sac_throttle"), False
    else:
        raise ValueError(f"unknown MODE: {MODE}")

    ray_ids = [-1] * NUM_RAYS # reused each frame so lidar lines don't pile up
    goal_ids = []
    obs, _ = env.reset()
    if MODE in ("nav", "throttle"):
        draw_track(env)
    elif MODE == "goal":
        goal_ids = draw_goal(env, goal_ids)

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                if MODE == "goal":
                    outcome = "reached" if info.get("success") else \
                              "collision" if info.get("collision") else "failed"
                    print(f"episode end: {outcome}  (dist {info.get('dist', 0):.2f} m, {env.steps} steps)")
                obs, _ = env.reset()
                if MODE == "goal":
                    goal_ids = draw_goal(env, goal_ids)

            if show_lidar:
                froms, hits, fracs = env.sim.raycast_points(num_rays=NUM_RAYS, max_range=LIDAR_RANGE)
                for k in range(NUM_RAYS):
                    color = [1, 0, 0] if fracs[k] < 1.0 else [0, 1, 0]  # red = hit, green = clear
                    ray_ids[k] = p.addUserDebugLine(froms[k], hits[k], color, 1, replaceItemUniqueId=ray_ids[k])

            pos, *_ = env.sim.observe()
            cam = p.getDebugVisualizerCamera()
            p.resetDebugVisualizerCamera(cam[10], cam[8], cam[9], [pos[0], pos[1], 0]) # distance, yaw, pitch, target
            time.sleep(env.sim.dt)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
