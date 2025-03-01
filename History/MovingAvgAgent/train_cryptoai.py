import argparse
from pathlib import Path
PARENT=Path(__file__).parent
import os
import datetime
import yaml
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import DummyVecEnv, Monitor

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent
from test_crypto import test_env

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--conf", type=str, default=None)
    args=parser.parse_args()

    if args.conf is None:
        conf_path=str(PARENT/"conf.yml")
        result_path=PARENT/"output"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S")/"result"
    else:
        conf_path=args.conf
        result_path=Path(args.conf).parent/"result"
    os.makedirs(result_path, exist_ok=True)
    with open(conf_path, "r", encoding="utf-8") as f:
        conf=yaml.safe_load(f)

    # 設定ファイルのコピー
    with open(result_path.parent/"conf.yml", "w", encoding="utf-8") as f:
        yaml.dump(conf, f, sort_keys=False, indent=4)

    T=conf["T"]
    moving_avg_windows=conf["moving_avg_windows"]
    alpha_realized=conf["alpha_realized"]
    alpha_unrealized=conf["alpha_unrealized"]
    beta_reward=conf["beta_reward"]
    total_timesteps=conf["total_timesteps"]
    model_name=str(result_path/conf["model_name"])
    n_env=conf["n_env"]

    # エージェントの設定
    agent=CryptoAgent()

    if args.train:

        # Parallel environments
        def make_env():
            env=CryptoEnv(
                T=T,
                moving_avg_windows=moving_avg_windows,
                alpha_realized=alpha_realized,
                alpha_unrealized=alpha_unrealized,
                beta_reward=beta_reward,
                agent=agent,
                datapath="data/train.csv"
            )
            env=Monitor(env, filename=None, allow_early_resets=True) #これでevalが出る
            return env
        
        vec_env=DummyVecEnv([make_env for _ in range(n_env)])
        model = PPO(MlpPolicy, vec_env, verbose=1)
        model.learn(total_timesteps=total_timesteps,progress_bar=True)
        model.save(model_name)

        del model # remove to demonstrate saving and loading


    # -- モデルのテスト --
    env=CryptoEnv(
        T=T,
        moving_avg_windows=moving_avg_windows,
        alpha_realized=alpha_realized,
        alpha_unrealized=alpha_unrealized,
        beta_reward=beta_reward,
        agent=agent,
        datapath="data/train.csv"
    )
    model = PPO.load(model_name)
    test_env(env, model, n_test=25, save_path=result_path)


if __name__ == "__main__":
    main()
