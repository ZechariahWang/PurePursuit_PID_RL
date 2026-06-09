import os, sys
sys.path.insert(0, os.path.join,(os.path.dirname(__file__), "src"))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from rl.car_env import CarSpeedEnv

def main():
    env = Monitor(CarSpeedEnv(gui=False, max_steps=1500))

    model = SAC("MLpPolicy". env, learning_rate=3e-4, buffer_size=200000, batch_size=256, gamma=0.99, tau=0.005, verbose=1, tensorboard_log="./tb")
    model.learn(total_timesteps=100000, progress_bar=True)
    model.save("sac_throttle")
    env.close()
    print("model saved")

if __name__=="__main__":
    main()