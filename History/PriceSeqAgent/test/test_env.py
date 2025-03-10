from pathlib import Path
ROOT=Path(__file__).parent.parent

import sys
sys.path.append(str(ROOT))

import numpy as np

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent


def main():

    agent=CryptoAgent()

    env=CryptoEnv(
        T=10,
        sequence_length=5,
        alpha_realized=0.1,
        alpha_unrealized=0.1,
        beta_reward=0.1,
        agent=agent,
        datapath=str(ROOT/"data/train.csv")
    )

    obs, _=env.reset()

    done=False
    while not done:
        action=np.random.randint(0, 4)
        obs, reward, done, _, _=env.step(action)


if __name__=="__main__":
    main()



