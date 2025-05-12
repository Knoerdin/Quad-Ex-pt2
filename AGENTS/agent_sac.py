# agent.py
import argparse
from stable_baselines3 import SAC
from env import QuadrupedEnv
import gymnasium as gym
from gym_custom_terrain import custom_make
import os
import pandas as pd

from stable_baselines3.common.env_checker import check_env

log_dir = "LOGS/TENSOR/"
models_dir = "TRAINED_MODELS/"

path_expands = ['_Ant-v5/', '_Quad-Ex/']

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)



def check_env(env):
    try:
        check_env(env, warn=True)
        print("Environment check passed.")
    except Exception as e:
        print(f"Environment check failed: {e}")

def train(model_path: str, total_timesteps: int, use_preset: bool = False):
    if use_preset:
        env = gym.make('Ant-v5', render_mode=None)
        path_extent = path_expands[0]
    else:
        path_extent = path_expands[1]
        raw_env = QuadrupedEnv(model_path, render_mode=None)
        check_env(raw_env)
        # dummy_env = DummyVecEnv([lambda: raw_env])
        # check_env(dummy_env)
        # env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True)

    model = SAC(policy="MlpPolicy", env=raw_env, verbose=1, tensorboard_log=log_dir)

    print(f"Started training using device: {model.device}")

    TIME_STEPS = total_timesteps
    for i in range(1,100):
        print(f"Training for {TIME_STEPS*i} steps")
        model.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name=f"SAC{path_extent}")
        model.save(f"{models_dir}SAC{path_extent}/steps={TIME_STEPS*i}")

    env.close()
    return model

def visualize(model_path: str, use_preset: bool = False, random_terrain: bool = False):
    if random_terrain and use_preset:
        print('random_terrain')
        env = custom_make('CustomTerrainAnt-v0', '../terrain_noise.png')
        mode_number = input("Enter model stepcount (SAC_quadex/steps=x*10000): ")
        model = SAC.load(f"{models_dir}SAC{path_expands[0]}steps={int(mode_number)*10000}", env=env)
    elif use_preset:
        env = gym.make('Ant-v5', render_mode="human")
        path = f'{models_dir}SAC{path_expands[0]}'
        mode_number = input(f"Enter model stepcount ({path}steps=x*10000): ")
        model = SAC.load(f"{path}steps={int(mode_number)*10000}", env=env)
    else:
        env = QuadrupedEnv(model_path, render_mode="human")
        path = f'{models_dir}SAC{path_expands[1]}'
        model_number = input(f"Enter model stepcount ({path}steps=x*10000): ")
        model = SAC.load(f"{path}steps={int(model_number)*10000}", env=env)
    distancs = []
    for ep in range(5):
        total_reward = 0
        obs, _ = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
        distance = info.get('distance_from_origin')
        distancs.append(distance)
        pd.DataFrame(distancs).to_csv("training_csv/distances_SAC_.csv")
        print(f'Total episode reward: {total_reward}, with a distance of: {distance}')
    env.close()

def main(training: bool = False, model_path: str = "models/quad_ex.xml", total_timesteps: int = 50_000, use_preset: bool = False, random_terrain: bool = False):
    if training:
        print("Training...")
        train(model_path=model_path, total_timesteps=total_timesteps, use_preset=use_preset)
    else:
        print("Visualizing...")
        visualize(model_path, use_preset, random_terrain=random_terrain)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train & visualize SAC on Quadruped")
    parser.add_argument("--model-path",     type=str, default="DEPENDENCIES/quad_ex.xml")
    parser.add_argument("--total-timesteps",type=int, default=10_000)
    parser.add_argument("--eval-steps",     type=int, default=1_000)
    parser.add_argument("--training", type=bool, default=False)
    parser.add_argument('--use-preset', type=bool, default=False)
    parser.add_argument('--random-terrain', type=bool, default=False)
    args = parser.parse_args()
    main(training=args.training, model_path=args.model_path, total_timesteps=args.total_timesteps, use_preset=args.use_preset, random_terrain=args.random_terrain)