from pathlib import Path
ROOT=Path(__file__).parent.parent

import sys
sys.path.append(str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent


def main():

    agent=CryptoAgent()

    t_past=30
    t_future=15
    env=CryptoEnv(
        trade_term=5,
        t_past=t_past,
        t_future=t_future,
        agent=agent,
        datapath=str(ROOT/"data/test.csv"),
        is_test_log=False
    )

    obs, _=env.reset()

    open_trj=[]
    idx_trj=[]

    done=False
    while not done:
        action=np.random.randint(0, 2)

        new_idx=[env.current_idx-(t_past-i+1) for i in range(t_past)]
        future_idx=env.current_idx+t_future
        idx_trj.append(np.concatenate([new_idx, [future_idx]]))
        future_data=env.debug_future_nrm_data()

        obs, reward, done, _, _=env.step(action)

        new_data=np.concatenate([
            obs.reshape(-1,5)[:,0],
            [future_data["close"]]
        ])
        
        open_trj.append(new_data) #open系列を集めていく

    open_trj=np.array(open_trj)
    idx_trj=np.array(idx_trj)

    open_trj=np.array(open_trj)
    print(open_trj)
    print(open_trj.shape)
    
    idx_trj=np.array(idx_trj)
    print(idx_trj)
    print(idx_trj.shape)

    colors = plt.get_cmap('tab10', open_trj.shape[0])  # 色マップを用意
    for i in range(open_trj.shape[0]):
        color=colors(i)

        if i==0:
            plt.plot(idx_trj[i][:-1], open_trj[i][:-1]+0.003*i, label="input sequence")
            plt.plot(idx_trj[i][-2:], open_trj[i][-2:]+0.003*i,linestyle="--",marker="x",markersize=6,color=color, label="future value")
        else:
            plt.plot(idx_trj[i][:-1], open_trj[i][:-1]+0.003*i)
            plt.plot(idx_trj[i][-2:], open_trj[i][-2:]+0.003*i,linestyle="--",marker="x",markersize=6,color=color)
    plt.legend()
    plt.savefig(str(Path(__file__).parent/"test_env_result.png"))


if __name__=="__main__":
    main()



