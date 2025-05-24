from pathlib import Path
ROOT=Path(__file__).parent.parent

import sys
sys.path.append(str(ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv, Monitor

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent
from src.transformer_policy import TransformerNetworkPolicy

def main():

    agent=CryptoAgent()

    def make_env():
        env=CryptoEnv(
            trade_term=10,
            sequence_length=6,
            agent=agent,
            datapath=str(ROOT/"data/test.csv")
        )
        env=Monitor(env, filename=None,allow_early_resets=True)
        return env

    env=DummyVecEnv([make_env for _ in range(1)])
    
    policy_kwargs={
        "network_kwargs":{
            "transformer_hidden_dim":512,
            "transformer_n_heads":8,
            "transformer_n_layers":6,
            "actor_dim":256,
            "actor_num_layers":2,
            "critic_dim":256,
            "critic_num_layers":2
        }
    }
    model=PPO(
        TransformerNetworkPolicy, 
        env, verbose=1,
        policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=1000,progress_bar=True)


if __name__=="__main__":
    main()



