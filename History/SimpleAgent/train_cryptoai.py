import argparse

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import DummyVecEnv, Monitor

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent
from test_crypto import test_env

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args=parser.parse_args()

    T=350
    alpha_realized=1
    alpha_unrealized=0.5
    agent=CryptoAgent()

    if args.train:

        # Parallel environments
        def make_env():
            env=CryptoEnv(
                T=T,
                alpha_realized=alpha_realized,
                alpha_unrealized=alpha_unrealized,
                agent=agent,
                datapath="data/train.csv"
            )
            env=Monitor(env, filename=None, allow_early_resets=True) #これでevalが出る
            return env
        
        vec_env=DummyVecEnv([make_env for _ in range(10)])
        model = PPO(MlpPolicy, vec_env, verbose=1)
        model.learn(total_timesteps=500000,progress_bar=True)
        model.save("crypto_ppo")

        del model # remove to demonstrate saving and loading


    # -- モデルのテスト --
    env=CryptoEnv(
        T=T,
        alpha_realized=alpha_realized,
        alpha_unrealized=alpha_unrealized,
        agent=agent,
        datapath="data/train.csv"
    )
    model = PPO.load("crypto_ppo_")
    test_env(env, model)


if __name__ == "__main__":
    main()
