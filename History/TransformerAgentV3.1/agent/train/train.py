from pathlib import Path
PARENT_DIR=Path(__file__).parent
MODEL_ROOT=Path(__file__).parent.parent
PROJECT_ROOT=MODEL_ROOT.parent.parent.parent
import sys
sys.path.append(str(MODEL_ROOT))
sys.path.append(str(MODEL_ROOT.parent))

import argparse
import yaml
from yaml import safe_load as yaml_safe_load
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv, Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch

from predictor.src.prob_transformer import ProbTransformer

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent
from src.cls_transformer_policy import ClsTransformerNetworkPolicy
from test_utils import test_env



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



def create_result_path(root:Path, model_name:str) -> Path:
    current_time=datetime.now().strftime("%Y%m%d_%H%M%S")
    return root/model_name/current_time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=PARENT_DIR/"conf.yml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train", action="store_true")
    args=parser.parse_args()

    device=args.device

    conf=yaml_safe_load(open(args.config,"r",encoding="utf-8"))
    conf_predictor=yaml_safe_load(open(
        PROJECT_ROOT/conf["predictor"]["confpath"],
        "r",encoding="utf-8"
    ))


    result_path=create_result_path(MODEL_ROOT/"outputs/",conf["train"]["model_name"])
    os.makedirs(result_path,exist_ok=True)

    # **************************************** configの保存 *************************************************
    with open(result_path/"config.yml","w",encoding="utf-8") as f:
        dump_conf=conf.copy()
        for key, value in conf_predictor.items():
            dump_conf["predictor"][key]=value
        yaml.dump(dump_conf,f,default_flow_style=False,encoding="utf-8",allow_unicode=True)


    # **************************************** agentの準備 *************************************************
    agent=CryptoAgent()


    # **************************************** envの準備 *************************************************
    trade_term=conf_predictor["model"]["input_params"]["s_past"]
    s_past=trade_term
    s_future=conf_predictor["model"]["input_params"]["s_future"]
    train_datapth=PROJECT_ROOT/conf_predictor["train"]["datapath"]["train"]
    test_datapth=PROJECT_ROOT/conf_predictor["train"]["datapath"]["test"]
    
    def make_train_env():
        env=CryptoEnv(
            trade_term=trade_term,
            s_past=s_past,
            s_future=s_future,
            agent=agent,
            datapath=train_datapth
        )

        env=Monitor(
            env,
            filename=None,
            allow_early_resets=True
        )
        return env
    
    def make_test_env():
        env=CryptoEnv(
            trade_term=trade_term,
            s_past=s_past,
            s_future=s_future,
            agent=agent,
            datapath=test_datapth
        )
        return env


    # **************************************** predictorのロード *************************************************
    predictor=ProbTransformer.from_config(conf_predictor["model"])
    predictor.load_state_dict(torch.load(
        PROJECT_ROOT/conf["predictor"]["modelpath"],
        map_location=device
    ))
    predictor.eval()
    predictor.to(device)
    predictor.requires_grad_(False) # モデルは凍結
    print("*"*50 + "predictor loaded" + "*"*50)
    print(predictor)
    print("*"*100)
    
    # **************************************** 学習の準備 *************************************************
    if args.train:

        # **************************************** チェックポイントの準備 *************************************************
        save_freq=conf["train"]["save_freq"]
        save_path=result_path/"checkpoints"
        name_prefix=conf["train"]["log_name"]
        n_test=conf["train"]["n_test"]

        checkpoint_callback=CheckpointAndTestCallback(
            test_env_fn=make_test_env,
            agent=agent,
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            n_test=n_test
        )


        # **************************************** 学習 *************************************************
        total_timesteps=conf["train"]["total_timesteps"]
        n_env=conf["train"]["n_env"]

        ## モデルのパラメータとpredictorを渡す
        policy_kwargs={}
        for key, value in conf["policy"].items():
            policy_kwargs[key]=value
        policy_kwargs["predictor"]=predictor
        policy_kwargs["s_future"]=s_future

        vec_env=DummyVecEnv([make_train_env for _ in range(n_env)])
        model = PPO(
            ClsTransformerNetworkPolicy, vec_env, verbose=1, 
            policy_kwargs={"network_kwargs":policy_kwargs},
            tensorboard_log=str(result_path/"log"),
        )

        model.learn( # 2048step/env貯まるとbackwardが走る
            total_timesteps=total_timesteps,
            progress_bar=True,tb_log_name=name_prefix,
            callback=checkpoint_callback
        )
        model.save(result_path/"model.zip")
        del model # remove to demonstrate saving and loading



    # **************************************** モデルのテスト *************************************************
    model=PPO.load(result_path/"model.zip")
    test_env(make_test_env(), model, n_test=50, save_path=result_path)



if __name__ == "__main__":
    main()