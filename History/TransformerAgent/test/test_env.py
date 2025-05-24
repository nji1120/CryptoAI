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
        trade_term=10,
        sequence_length=5,
        agent=agent,
        datapath=str(ROOT/"data/test.csv"),
        is_test_log=True
    )

    obs, _=env.reset()

    done=False
    while not done:
        action=np.random.randint(0, 2)
        obs, reward, done, _, _=env.step(action)


if __name__=="__main__":
    main()



