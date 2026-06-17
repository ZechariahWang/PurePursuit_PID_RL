import os, sys, shutil
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from rl.environments.goal_nav_env import GoalNavEnv

SEED = 0
MAX_STEPS = 1000

# Remove-Item -Recurse -Force .\checkpoints -ErrorAction SilentlyContinue
# Remove-Item -Force .\sac_goal_nav*.zip -ErrorAction SilentlyContinue

WARMUP_STEPS   = 50_000
MAIN_STEPS     = 350_000
OBSTACLE_RANGE = (3, 7) 

def make_env(seed, n_obstacles=0, obstacle_range=None):
    env = Monitor(GoalNavEnv(gui=False, max_steps=MAX_STEPS,
                             n_obstacles=n_obstacles, obstacle_range=obstacle_range))
    env.reset(seed=seed)
    return env

def main():
    set_random_seed(SEED)

    env = make_env(SEED, n_obstacles=0)
    model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000, batch_size=256,
                gamma=0.99, tau=0.005, verbose=1, tensorboard_log="./tb", seed=SEED)

    # phase 1: empty-world, just wanna try and get to the goal
    print(f"\n=== warmup: no obstacles, {WARMUP_STEPS} steps ===")
    eval_env = make_env(SEED + 1000, n_obstacles=0)
    model.learn(total_timesteps=WARMUP_STEPS, progress_bar=True, tb_log_name="SAC_curriculum",
                callback=[CheckpointCallback(save_freq=25000, save_path="./checkpoints", name_prefix="sac_goal_nav_warmup"),
                          EvalCallback(eval_env, best_model_save_path="./checkpoints/best/warmup",
                                       log_path="./checkpoints/eval/warmup", eval_freq=25000,
                                       n_eval_episodes=5, deterministic=True)])
    eval_env.close()

    # phase 2: randomized obstacle count
    print(f"\n=== main: obstacles randomized in {OBSTACLE_RANGE}, {MAIN_STEPS} steps ===")
    main_env = make_env(SEED + 1, obstacle_range=OBSTACLE_RANGE)
    model.set_env(main_env)
    env.close()
    env = main_env
    eval_env = make_env(SEED + 1001, obstacle_range=OBSTACLE_RANGE)
    model.learn(total_timesteps=MAIN_STEPS, progress_bar=True, reset_num_timesteps=False,
                tb_log_name="SAC_curriculum",
                callback=[CheckpointCallback(save_freq=25000, save_path="./checkpoints", name_prefix="sac_goal_nav_main"),
                          EvalCallback(eval_env, best_model_save_path="./checkpoints/best/main",
                                       log_path="./checkpoints/eval/main", eval_freq=25000,
                                       n_eval_episodes=10, deterministic=True)])
    eval_env.close()
    env.close()

    # use the best eval checkpoint, not the final model
    best = "./checkpoints/best/main/best_model.zip"
    if os.path.exists(best):
        shutil.copy(best, "sac_goal_nav.zip")
        print(f"saved best eval model at sac_goal_nav.zip  (from {best})")
    else:
        model.save("sac_goal_nav")
        print("no best checkpoint found; saved final model at sac_goal_nav.zip")

    # archive copy of model
    os.makedirs("models/archive", exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_path = os.path.join("models/archive", f"{stamp}_sac_goal_nav.zip")
    shutil.copy("sac_goal_nav.zip", archive_path)
    print(f"archived -> {archive_path}")

if __name__ == "__main__":
    main()
