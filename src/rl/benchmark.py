import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from rl.environments.goal_nav_env import GoalNavEnv, MAX_STEER
from pure_pursuit import steering_angle
from pid import PID

TARGET_SPEED = 3.0
N_EPISODES = 30

def rl_controller(model):
    def act(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return act

def baseline_controller():
    """Pure pursuit aimed straight at the goal + PID throttle.

    Purely geometric: it steers at the target and has no idea obstacles exist,
    so it serves as the 'no avoidance' reference the RL policy is compared to.
    """
    pid = PID(kp=0.5, ki=0.1, kd=0.0)
    def act(env, obs):
        pos, yaw, speed = env.sim.observe()
        Ld = max(float(np.hypot(*(env.goal - pos))), 1.0)  # aim at the goal point
        steer = steering_angle(pos, yaw, env.goal, env.sim.WHEELBASE, Ld)
        steer_norm = float(np.clip(steer / MAX_STEER, -1, 1))
        throttle = float(pid.step(TARGET_SPEED, speed, env.sim.dt))
        return np.array([throttle, steer_norm], dtype=np.float32)
    return act, pid

def run(env, controller, pid=None, n=N_EPISODES):
    st = {"final_dist": [], "steps": [], "collisions": 0, "reached": 0, "failed": 0}
    for s in range(n):
        if pid is not None:
            pid.reset()
        obs, _ = env.reset(seed=s)
        info, done = {}, False
        while not done:
            obs, r, term, trunc, info = env.step(controller(env, obs))
            done = term or trunc
        st["final_dist"].append(info.get("dist", 0.0))
        st["steps"].append(env.steps)
        if info.get("success"):
            st["reached"] += 1
        elif info.get("collision"):
            st["collisions"] += 1
        else:
            st["failed"] += 1   # off-bounds or timeout
    return st

def summarize(name, st):
    n = len(st["final_dist"])
    print(f"\n=== {name} ===")
    print(f"  reached goal  : {st['reached']:>3}/{n}")
    print(f"  collisions    : {st['collisions']:>3}/{n}")
    print(f"  failed/timeout: {st['failed']:>3}/{n}")
    print(f"  mean final d  : {np.mean(st['final_dist']):6.2f} m")
    print(f"  mean steps    : {np.mean(st['steps']):6.0f}")

def main():
    env = GoalNavEnv(gui=False)
    model = SAC.load("sac_goal_nav")

    rl_act = rl_controller(model)
    base_act, pid = baseline_controller()

    summarize("RL (sac_goal_nav)", run(env, rl_act))
    summarize("Pure pursuit + PID (no avoidance)", run(env, base_act, pid=pid))
    env.close()

if __name__ == "__main__":
    main()
