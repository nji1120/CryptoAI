from pathlib import Path
ROOT=Path(__file__).parent.parent

import sys
sys.path.append(str(ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv, Monitor

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent
from src.lstm_policy import LSTMNetworkPolicy

def main():

    agent=CryptoAgent()

    def make_env():
        env=CryptoEnv(
            trade_term=10,
            sequence_length=5,
            agent=agent,
            datapath=str(ROOT/"data/test.csv")
        )
        env=Monitor(env, filename=None,allow_early_resets=True)
        return env

    env=DummyVecEnv([make_env for _ in range(1)])
    
    policy_kwargs={
        "network_kwargs":{
            "lstm_hidden_dim":64,
            "lstm_num_layers":1,
            "actor_dim":128,
            "critic_dim":128
        }
    }
    model=PPO(
        LSTMNetworkPolicy, 
        env, verbose=1,
        policy_kwargs=policy_kwargs
        )
    model.learn(total_timesteps=1000,progress_bar=True)


if __name__=="__main__":
    main()



