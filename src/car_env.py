import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import make_track, nearest_index, lookahead_point
from pure_pursuit import steering_angle
from sim import CarSim

WAYPOINTS = [(0, 0), (30, 5), (40, 25), (20, 40), (-10, 30), (-20, 5)]
LD = 2.0      
MAX_STEER = 0.6   

class CarSpeedEnv(gym.Env):
    metadata={"render_modes": ["human"]}

    def __init__(self, gui=False, dt=1/60, target_speed_range=(1, 4), max_steps=1500, jerk_penalty=0.02):
        super().__init__()
        self.sim = CarSim(gui=gui, dt=dt)
        self.track = make_track(WAYPOINTS, samples_per_seg=200)
        self.target_speed_range = target_speed_range
        self.max_steps = max_steps
        self.jerk_penalty = jerk_penalty
        self.speed_scale = 10

        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space=spaces.Box(low=np.array([-2, 0, -2], dtype=np.float32), high=np.array([2, 1, 2], dtype=np.float32), dtype=np.float32)
        self.target_speed = 0
        self.prev_throttle = 0
        self.steps = 0

    def obs(self, speed):
        error = self.target_speed - speed
        return np.array([speed / self.speed_scale, self.target_speed / self.speed_scale, error / self.speed_scale], dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = self.track[0]
        heading = np.arctan2(self.track[1, 1] - self.track[0, 1], self.track[1,0] - self.track[0,0])
        self.sim_reset(pose=(start[0], start[1], heading))

        self.target_speed = float(self.np_random.uniform(*self.target_speed_range))
        self.prev_throttle = 0
        self.steps = 0
        _,_,speed = self.sim.observe()
        return self.obs(speed), {}
    
    def step(self, action):
        throttle = float(np.clip(action[0], -1, 1))
        pos, yaw, speed = self.sim.observe()

        index = nearest_index(self.track, pos)
        look_point, _ = lookahead_point(self.track, pos, LD, index)

        steer = steering_angle(pos, yaw, look_point, self.sim.WHEELBASE, LD)
        steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))

        self.sim.apply(steer, throttle)
        _,_, speed = self.sim.observe()

        self.steps += 1

        speed_error = abs(self.target_speed - speed)
        jerk = abs(throttle - self.prev_throttle)
        reward = -speed_error - self.jerk_penalty * jerk
        self.prev_throttle = throttle

        terminated = False
        truncated = self.steps >= self.max_steps
        return self.obs(speed), reward, terminated, truncated, {"speed" : speed}


