import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from rl.new_car_env import CarSpeedEnv, MAX_STEER
from track import nearest_index, lookahead_point
from pure_pursuit import steering_angle
from pid import PID

LD = 2.0
TARGET_SPEED = 3.0
N_EPISODES = 30

def rl_controller(model):
    def act(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return act

def baseline_controller():
    pid = PID(kp=0.5, ki=0.1, kd=0.0)
    def act(env, obs):
        pos, yaw, speed = env.sim.observe()
        idx = nearest_index(env.track, pos)
        look_pt, _ = lookahead_point(env.track, pos, LD, idx)
        steer = steering_angle(pos, yaw, look_pt, env.sim.WHEELBASE, LD)
        steer_norm = float(np.clip(steer / MAX_STEER, -1, 1))
        throttle = float(pid.step(TARGET_SPEED, speed, env.sim.dt))
        return np.array([throttle, steer_norm], dtype=np.float32)
    return act, pid

def run(env, controller, pid=None, n=N_EPISODES):
    st = {"progress": [], "steps": [], "collisions": 0, "offtrack": 0, "finished": 0}
    for s in range(n):
        if pid is not None:
            pid.reset()
        obs, _ = env.reset(seed=s)
        info, term, done = {}, False, False
        while not done:
            obs, r, term, trunc, info = env.step(controller(env, obs))
            done = term or trunc
        st["progress"].append(info.get("progress", 0.0))
        st["steps"].append(env.steps)
        if info.get("collision"):
            st["collisions"] += 1
        elif term:
            st["offtrack"] += 1
        else:
            st["finished"] += 1
    return st

def summarize(name, st):
    n = len(st["progress"])
    print(f"\n=== {name} ===")
    print(f"  mean progress : {np.mean(st['progress']):6.1f} m")
    print(f"  mean steps    : {np.mean(st['steps']):6.0f}")
    print(f"  collisions    : {st['collisions']:>3}/{n}")
    print(f"  off-track     : {st['offtrack']:>3}/{n}")
    print(f"  survived      : {st['finished']:>3}/{n}")

def main():
    env = CarSpeedEnv(gui=False)
    model = SAC.load("sac_nav")

    rl_act = rl_controller(model)
    base_act, pid = baseline_controller()

    summarize("RL (sac_nav)", run(env, rl_act))
    summarize("Pure pursuit + PID", run(env, base_act, pid=pid))
    env.close()

if __name__ == "__main__":
    main()
