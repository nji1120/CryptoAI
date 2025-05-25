import argparse
from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent.parent
import os
import datetime
import yaml
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv, Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent
from test_utils import test_env
from src.cls_transformer_policy import ClsTransformerNetworkPolicy


class CheckpointAndTestCallback(BaseCallback):
    """
    チェックポイントごとにテスト用環境を作成し, テストを行う.
    テスト結果をチェックポイントごとに保存する.
    """

    def __init__(self, test_env_fn, agent, save_freq, save_path, name_prefix, n_test, verbose=0):
        super().__init__(verbose)
        self.test_env_fn = test_env_fn
        self.agent = agent
        self.n_test = n_test
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.n_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps // self.save_freq >= self.n_save:
            self.n_save += 1

            checkpoint_dir = self.save_path / f"{self.name_prefix}_cp{self.num_timesteps}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            env = self.test_env_fn()
            self.model.save(checkpoint_dir / "model.zip")
            test_env(env, self.model, n_test=self.n_test, save_path=checkpoint_dir)
        return True


def get_policy_class(archi_type:str)->Callable:
    if archi_type.casefold()=="cls_transformer":
        return ClsTransformerNetworkPolicy
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
    t_past=train_conf["t_past"]
    t_future=train_conf["t_future"]
    total_timesteps=train_conf["total_timesteps"]
    model_name=str(result_path/train_conf["model_name"])
    n_env=train_conf["n_env"]
    log_name=train_conf["log_name"]
    save_freq=train_conf["save_freq"]
    n_test=train_conf["n_test"]

    if is_full_path(train_conf["datapath"]["train"]):
        data_path=train_conf["datapath"]["train"]
    else:
        data_path=ROOT/train_conf["datapath"]["train"]


    # エージェントの設定
    agent=CryptoAgent()

    # 環境の設定
    def make_env():
        env=CryptoEnv(
            trade_term=trade_term,
            t_past=t_past,
            t_future=t_future,
            agent=agent,
            datapath=data_path
        )
        env=Monitor(
            env, 
            filename=None, 
            allow_early_resets=True
        ) #これでevalが出る
        return env
    

    def make_test_env(): # monitorなしのtest_env
        env=CryptoEnv(
            trade_term=trade_term,
            t_past=t_past,
            t_future=t_future,
            agent=agent,
            datapath=data_path
        )
        return env

    if args.train:


        checkpoint_callback=CheckpointAndTestCallback(
            test_env_fn=make_test_env,
            agent=agent,
            save_freq=save_freq,
            save_path=result_path/"checkpoints",
            name_prefix=log_name,
            n_test=n_test
        )


        # Parallel environments        
        vec_env=DummyVecEnv([make_env for _ in range(n_env)])
        archi_type=model_conf["archi_type"]
        policy_class=get_policy_class(archi_type)
        model = PPO(
            policy_class, vec_env, verbose=1, 
            policy_kwargs=model_conf,
            tensorboard_log=str(result_path/"log"),
        )

        # 2048step/env貯まるとbackwardが走る
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,tb_log_name=log_name,
            callback=checkpoint_callback
        )
        model.save(model_name)

        del model # remove to demonstrate saving and loading


    # -- モデルのテスト --
    if is_full_path(train_conf["datapath"]["test"]):
        datapath=train_conf["datapath"]["test"]
    else:
        datapath=ROOT/train_conf["datapath"]["test"]
    env=CryptoEnv(
        trade_term=trade_term,
        t_past=t_past,
        t_future=t_future,
        agent=agent,
        datapath=datapath
    )
    model = PPO.load(model_name)
    test_env(env, model, n_test=50, save_path=result_path)


if __name__ == "__main__":
    main()
