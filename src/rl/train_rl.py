import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from rl.environments.goal_nav_env import GoalNavEnv

SEED = 0
MAX_STEPS = 1000

# curriculum: (n_obstacles, timesteps). Learn to drive to the goal in an empty
# world first, then progressively add obstacles to the same model.
STAGES = [
    (0, 100_000),
    (3, 100_000),
    (5, 100_000),
]

def make_env(n_obstacles, seed):
    env = Monitor(GoalNavEnv(gui=False, max_steps=MAX_STEPS, n_obstacles=n_obstacles))
    env.reset(seed=seed)
    return env

def main():
    set_random_seed(SEED)

    first_n, _ = STAGES[0]
    env = make_env(first_n, SEED)
    model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000, batch_size=64,
                gamma=0.99, tau=0.005, verbose=1, tensorboard_log="./tb", seed=SEED)

    for i, (n_obstacles, steps) in enumerate(STAGES):
        print(f"\n=== stage {i}: n_obstacles={n_obstacles}, {steps} steps ===")
        if i > 0:
            new_env = make_env(n_obstacles, SEED + i)   # swap difficulty, keep the model + replay buffer
            model.set_env(new_env)
            env.close()
            env = new_env

        # eval at this stage's difficulty so success numbers reflect the task it's on
        eval_env = make_env(n_obstacles, SEED + 1000 + i)
        tag = f"stage{i}_obs{n_obstacles}"
        checkpoint_cb = CheckpointCallback(save_freq=25000, save_path="./checkpoints",
                                           name_prefix=f"sac_goal_nav_{tag}")
        eval_cb = EvalCallback(eval_env, best_model_save_path=f"./checkpoints/best/{tag}",
                               log_path=f"./checkpoints/eval/{tag}", eval_freq=25000,
                               n_eval_episodes=5, deterministic=True)

        # reset_num_timesteps only on the first stage so the timestep counter and
        # tensorboard run continue unbroken across the whole curriculum
        model.learn(total_timesteps=steps, progress_bar=True,
                    callback=[checkpoint_cb, eval_cb],
                    reset_num_timesteps=(i == 0), tb_log_name="SAC_curriculum")
        model.save(f"sac_goal_nav_{tag}")
        eval_env.close()

    model.save("sac_goal_nav")
    env.close()
    print("model saved -> sac_goal_nav.zip")

if __name__ == "__main__":
    main()
