import argparse
from pathlib import Path
PARENT=Path(__file__).parent
import os
import datetime
import yaml
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv, Monitor

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent
from test_utils import test_env
from src.transformer_policy import TransformerNetworkPolicy
from src.sequence_transformer_policy import SequenceTransformerNetworkPolicy


def get_policy_class(archi_type:str)->Callable:
    if archi_type.casefold()=="transformer":
        return TransformerNetworkPolicy
    elif archi_type.casefold()=="sequence_transformer":
        return SequenceTransformerNetworkPolicy
    else:
        raise ValueError(f"Invalid architecture type: {archi_type}")

def is_full_path(path)-> bool:
    return os.path.isabs(path)

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


    # コメントごとconfのコピー
    with open(conf_path, "r", encoding="utf-8") as f:
        conf_txt=f.readlines() #コメントごと全部コピー
    with open(result_path.parent/"conf.yml", "w", encoding="utf-8") as f:
        f.writelines(conf_txt) #コメントごと全部ペースト


    train_conf, model_conf=conf["train"], conf["model"]
    trade_term=train_conf["trade_term"]
    sequence_length=train_conf["sequence_length"]
    total_timesteps=train_conf["total_timesteps"]
    model_name=str(result_path/train_conf["model_name"])
    n_env=train_conf["n_env"]


    if is_full_path(train_conf["datapath"]["train"]):
        data_path=train_conf["datapath"]["train"]
    else:
        data_path=PARENT/"data"/train_conf["datapath"]["train"]

    def make_env():
        env=CryptoEnv(
            trade_term=trade_term,
            sequence_length=sequence_length,
            agent=agent,
            datapath=data_path
        )
        env=Monitor(
            env, 
            filename=None, 
            allow_early_resets=True
        ) #これでevalが出る
        return env


    # エージェントの設定
    agent=CryptoAgent()

    if args.train:

        # Parallel environments        
        vec_env=DummyVecEnv([make_env for _ in range(n_env)])
        archi_type=model_conf["archi_type"]
        policy_class=get_policy_class(archi_type)
        model = PPO(
            policy_class, vec_env, verbose=1, 
            policy_kwargs=model_conf,
            tensorboard_log=str(result_path/"log")
        )
        model.learn(total_timesteps=total_timesteps,progress_bar=True)
        model.save(model_name)

        del model # remove to demonstrate saving and loading


    # -- モデルのテスト --
    if is_full_path(train_conf["datapath"]["test"]):
        datapath=train_conf["datapath"]["test"]
    else:
        datapath=PARENT/"data"/train_conf["datapath"]["test"]
    env=CryptoEnv(
        trade_term=trade_term,
        sequence_length=sequence_length,
        agent=agent,
        datapath=datapath
    )
    model = PPO.load(model_name)
    test_env(env, model, n_test=50, save_path=result_path)


if __name__ == "__main__":
    main()
