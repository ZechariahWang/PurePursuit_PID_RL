import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from track import make_track, nearest_index, cumulative_arclength, track_heading, cross_track_error
from sim import CarSim

WAYPOINTS           = [(0, 0), (30, 5), (40, 25), (20, 40), (-10, 30), (-20, 5)]
MAX_STEER           = 0.6
NUM_RAYS            = 16
LIDAR_RANGE         = 10.0
CTE_MAX             = 4.0
N_OBSTACLES         = 6

class CarSpeedEnv(gym.Env):
    metadata= {"render_modes" : ["human"]}

    def __init__(self, gui=False, dt = 1/60, max_steps = 2000, jerk_penalty=0.02, progress_weight=1, time_penalty=0.01, collision_penalty=20, offtrack_penalty=20):
        super().__init__()

        self.sim = CarSim(gui=gui, dt=dt)
        self.track = make_track(WAYPOINTS, samples_per_seg=200)
        self.arclen = cumulative_arclength(self.track)
        self.total_len = self.arclen[-1] + np.linalg.norm(self.track[0] - self.track[-1])
        self.max_steps = max_steps
        self.jerk_penalty = jerk_penalty
        self.progress_weight = progress_weight
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty
        self.offtrack_penalty = offtrack_penalty
        self.speed_scale = 10.0

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)   # [throttle, steer]
        low  = np.array([-2, -1, -1] + [0.0] * NUM_RAYS, dtype=np.float32)
        high = np.array([ 2,  1,  1] + [1.0] * NUM_RAYS, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.prev_throttle = 0.0
        self.prev_steer = 0.0
        self.prev_idx = 0
        self.steps = 0

    def obs(self):
        pos, yaw, speed = self.sim.observe()
        idx = nearest_index(self.track, pos)
        th = track_heading(self.track, idx)
        heading_err = np.arctan2(np.sin(th - yaw), np.cos(th - yaw))
        cte = cross_track_error(self.track, pos, idx)
        rays = self.sim.raycast(num_rays=NUM_RAYS, max_range=LIDAR_RANGE)
        head = np.array([speed / self.speed_scale,
                        heading_err / np.pi,
                        np.clip(cte / CTE_MAX, -1, 1)], dtype=np.float32)
        return np.concatenate([head, rays]).astype(np.float32), idx, cte

    def sample_obstacles(self):
        idxs = self.np_random.integers(40, len(self.track) - 5, size=N_OBSTACLES)
        positions = []
        for i in idxs:
            th = track_heading(self.track, int(i))
            nx, ny = -np.sin(th), np.cos(th)
            offset = self.np_random.uniform(-1.0, 1.0)
            positions.append((self.track[i][0] + nx * offset, self.track[i][1] + ny * offset))
        return positions
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start = self.track[0]
        heading = np.arctan2(self.track[1, 1] - self.track[0, 1], self.track[1, 0] - self.track[0, 0])
        self.sim.reset(pose=(start[0], start[1], heading))
        self.sim.spawn_obstacles(self._sample_obstacles())

        self.prev_throttle = 0.0
        self.prev_steer = 0.0
        self.steps = 0
        observation, idx, _ = self.obs()
        self.prev_idx = idx
        return observation, {}
    
    def step(self, action):
        throttle = float(np.clip(action[0], -1, 1))
        steer = float(np.clip(action[1], -1, 1)) * MAX_STEER
        self.sim.apply(steer, throttle)
        self.steps += 1

        observation, idx, cte = self.obs()

        ds = self.arclen[idx] - self.arclen[self.prev_idx]
        if ds < -self.total_len / 2: # wrapped forward over the seam
            ds += self.total_len
        elif ds > self.total_len / 2: # wrapped backward
            ds -= self.total_len
        self.prev_idx = idx

        jerk = abs(throttle - self.prev_throttle) + abs(steer - self.prev_steer)
        reward = self.progress_weight * ds - self.time_penalty - self.jerk_penalty * jerk
        self.prev_throttle = throttle
        self.prev_steer = steer

        terminated = False
        if self.sim.check_collision():
            reward -= self.collision_penalty
            terminated = True
        elif abs(cte) > CTE_MAX:
            reward -= self.offtrack_penalty
            terminated = True

        truncated = self.steps >= self.max_steps
        _, _, speed = self.sim.observe()
        return observation, reward, terminated, truncated, {"speed": speed}

    def close(self):
        try:
            p.disconnect(self.sim.client)
        except Exception:
            pass

    