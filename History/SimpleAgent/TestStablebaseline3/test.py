import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env('CartPole-v1', n_envs=10)

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=5000,progress_bar=True)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

env= make_vec_env('CartPole-v1', n_envs=1)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human') #引数にmode=humanを指定しないと何も描画されない