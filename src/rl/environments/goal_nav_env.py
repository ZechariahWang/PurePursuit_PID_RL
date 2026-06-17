import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from sim import CarSim

MAX_STEER    = 0.6
NUM_RAYS     = 16
LIDAR_RANGE  = 10.0

class GoalNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui=False, dt=1/60, max_steps=1500,
                 goal_dist_range=(8.0, 14.0), goal_radius=1.0,
                 n_obstacles=5, obstacle_range=None, obstacle_spread=2.0,
                 progress_weight=5.0, reach_bonus=50.0,
                 collision_penalty=15.0, oob_penalty=10.0,
                 time_penalty=0.02, jerk_penalty=0.0,
                 clearance_penalty=0.3, clearance_thresh=0.2):
        super().__init__()

        self.sim = CarSim(gui=gui, dt=dt)
        self.start = np.array([0.0, 0.0])
        self.max_steps = max_steps
        self.goal_dist_range = goal_dist_range
        self.goal_radius = goal_radius
        self.n_obstacles = n_obstacles
        self.obstacle_range = obstacle_range  # if set (lo, hi), sample count each reset
        self.obstacle_spread = obstacle_spread
        self.progress_weight = progress_weight
        self.reach_bonus = reach_bonus
        self.collision_penalty = collision_penalty
        self.oob_penalty = oob_penalty
        self.time_penalty = time_penalty
        self.jerk_penalty = jerk_penalty
        self.clearance_penalty = clearance_penalty   # nudge away from obstacles before contact
        self.clearance_thresh = clearance_thresh     # in ray-fraction units (0.2 -> within 20% of lidar range)
        self.speed_scale = 10.0
        self.dist_scale = 15.0
        self.bound_margin = 10.0 # give up past this distance

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)  # [throttle, steer]
        # obs = [speed, dist_to_goal, sin(bearing), cos(bearing)] + lidar rays
        low  = np.array([-2, 0, -1, -1] + [0.0] * NUM_RAYS, dtype=np.float32)
        high = np.array([ 2, 2,  1,  1] + [1.0] * NUM_RAYS, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.goal = self.start.copy()
        self.prev_dist = 0.0
        self.init_dist = 0.0
        self.prev_throttle = 0.0
        self.prev_steer = 0.0
        self.steps = 0

    def sample_goal(self):
        ang = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(*self.goal_dist_range)
        goal = self.start + dist * np.array([np.cos(ang), np.sin(ang)])
        return goal, ang

    # scatter random obstacles around
    def sample_obstacles(self, goal):
        seg = goal - self.start
        perp = np.array([-seg[1], seg[0]]) / (np.linalg.norm(seg) + 1e-9)
        positions, tries = [], 0
        while len(positions) < self.n_obstacles and tries < 100:
            tries += 1
            t = self.np_random.uniform(0.35, 0.85)  # leave clear runway so it learns to drive first
            off = self.np_random.uniform(-self.obstacle_spread, self.obstacle_spread)
            pt = self.start + t * seg + perp * off
            if np.linalg.norm(pt - self.start) < 2.0: # dont spawn blocks on starting loc
                continue
            if np.linalg.norm(pt - goal) < self.goal_radius + 1.0: # keep goal 
                continue
            positions.append((pt[0], pt[1]))
        return positions

    def obs(self):
        pos, yaw, speed = self.sim.observe()
        dx, dy = self.goal[0] - pos[0], self.goal[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        # rotate the goal vector into the car's body frame -> bearing relative to heading
        local_x =  np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        bearing = np.arctan2(local_y, local_x)
        rays = self.sim.raycast(num_rays=NUM_RAYS, max_range=LIDAR_RANGE)
        head = np.array([speed / self.speed_scale,
                         min(dist / self.dist_scale, 2.0),
                         np.sin(bearing), np.cos(bearing)], dtype=np.float32)
        return np.concatenate([head, rays]).astype(np.float32), dist

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.obstacle_range is not None:   # randomize difficulty so easy + hard layouts are always seen
            lo, hi = self.obstacle_range
            self.n_obstacles = int(self.np_random.integers(lo, hi + 1))
        self.goal, goal_ang = self.sample_goal()

        # start roughly facing the goal 
        heading = goal_ang + self.np_random.uniform(-0.3, 0.3)
        self.sim.reset(pose=(self.start[0], self.start[1], heading))
        self.sim.create_obstacles(self.sample_obstacles(self.goal))

        self.prev_throttle = 0.0
        self.prev_steer = 0.0
        self.steps = 0
        observation, dist = self.obs()
        self.prev_dist = dist
        self.init_dist = dist
        return observation, {}

    def step(self, action):
        throttle = float(np.clip(action[0], -1, 1))
        steer = float(np.clip(action[1], -1, 1)) * MAX_STEER
        self.sim.apply(steer, throttle)
        self.steps += 1

        observation, dist = self.obs()

        # reward closing distance to the goal
        reward = self.progress_weight * (self.prev_dist - dist) - self.time_penalty
        jerk = abs(throttle - self.prev_throttle) + abs(steer - self.prev_steer)
        reward -= self.jerk_penalty * jerk
        # continuous clearance penalty: discourage getting close to obstacles so it
        # detours early instead of nosing in and oscillating (rays: 1=clear, low=close)
        min_ray = float(observation[4:].min())
        reward -= self.clearance_penalty * max(0.0, self.clearance_thresh - min_ray)
        self.prev_dist = dist
        self.prev_throttle = throttle
        self.prev_steer = steer

        terminated = False
        info = {"dist": dist, "success": False, "collision": False}

        if self.sim.check_collision():
            reward -= self.collision_penalty
            terminated = True
            info["collision"] = True
        elif dist < self.goal_radius:
            reward += self.reach_bonus
            terminated = True
            info["success"] = True
        elif dist > self.init_dist + self.bound_margin:   # wandered off
            reward -= self.oob_penalty
            terminated = True

        truncated = self.steps >= self.max_steps
        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            p.disconnect(self.sim.client)
        except Exception:
            pass
