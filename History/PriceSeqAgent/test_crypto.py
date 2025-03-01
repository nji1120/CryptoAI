import os

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from stable_baselines3 import PPO
from src.crypto_env import CryptoEnv


def test_env(env:CryptoEnv, policy:PPO, n_test:int=1, save_path:Path=None):
    
    results=[]

    print(f"\033[32m--- START test_env "+"-"*50+"\033[0m")
    for i_test in tqdm(range(n_test), desc="test_env"):

        obs_history=[]
        reward_history=[]
        action_history=[]
        realized_pnl_rate_history=[] # 累積確定利益率

        obs,_=env.reset()
        done=False
        while not done:
            action, _states=policy.predict(obs, deterministic=False)
            action_history.append(
                env.agent.actual_action(action).value
                # action
            )
            obs, reward, done,_,info=env.step(action)

            obs_history.append(obs[-8:-3])
            reward_history.append(reward)
            realized_pnl_rate_history.append(
                env.agent.cumlative_realized_pnl_rate
            )

        results.append({
            "i": i_test,
            "r_mean": np.mean(reward_history),
            "r_std": np.std(reward_history),
            "r_total": sum(reward_history),
            "pnl_mean": np.mean(realized_pnl_rate_history),
            "pnl_std": np.std(realized_pnl_rate_history),
            "pnl_last": realized_pnl_rate_history[-1],
        })

        visualize_history(
            obs_history, action_history, 
            reward_history, realized_pnl_rate_history, 
            save_path/"test_imgs", 
            filename=f"result_{i_test}.png"
        )

    results_df=pd.DataFrame(results)
    results_df.to_csv(save_path/"results.csv", index=False)
    visualize_results(results_df, save_path)

    print(f"\033[32m--- END test_env --"+"-"*50+"\033[0m")


def visualize_history(
        obs_history, action_history, 
        reward_history, realized_pnl_rate_history, 
        save_path:Path=None, filename:str="test.png"
    ):
    # Convert obs_history to a NumPy array for easier indexing
    obs_array = np.array(obs_history)
    reward_array = np.array(reward_history)
    realized_pnl_rate_array = np.array(realized_pnl_rate_history)

    # Plot OHLCV data
    fig, axs=plt.subplots(3, 1, figsize=(6,6))

    # Plot Open, High, Low, Close
    axs[0].plot(obs_array[:, 0], label='Open', color='blue', alpha=0.3)
    axs[0].plot(obs_array[:, 1], label='High', color='green', alpha=0.3)
    axs[0].plot(obs_array[:, 2], label='Low', color='red', alpha=0.3)
    axs[0].plot(obs_array[:, 3], label='Close', color='orange', alpha=1)
    axs[0].set_title('OHLC Data')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # Plot Volume
    axs[1].plot(obs_array[:, 4], label='Volume', color='purple')
    axs[1].set_title('Volume')
    axs[1].set_ylabel('Volume')
    axs[1].legend()

    # Plot Actions
    markers=["o", "x", "s", "D"]
    colors=["red", "blue", "green", "gray"]
    labels=["Long", "Short", "Close", "Stay"]
    for i_action in range(4):
        idx=np.where(np.array(action_history)==i_action)[0]
        axs[0].plot(
            idx, obs_array[idx, 3], markers[i_action], 
            label=labels[i_action], color=colors[i_action],
            markersize=4
        )  # Plot on Close


    # Plot Immediate Rewards
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('Immediate Reward', color='cyan')
    axs[2].plot(reward_array, label='Immediate Reward', color='cyan')
    axs[2].tick_params(axis='y', labelcolor='cyan')

    # Create a second y-axis for Cumulative Reward
    ax2 = axs[2].twinx()
    ax2.set_ylabel('Realized PNL Rate', color='magenta')
    ax2.plot(realized_pnl_rate_array, label='Realized PNL Rate', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')

    axs[2].set_title('Rewards / Realized PNL Rate')

    for ax in axs:
        ax.grid()

    plt.tight_layout()

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path/filename)
        plt.close()
    else:
        plt.show()


def visualize_results(results_df: pd.DataFrame, save_path: Path = None):
    r_mean = results_df["r_mean"]
    pnl_mean = results_df["pnl_mean"]
    pnl_last = results_df["pnl_last"]

    fig, ax1 = plt.subplots()

    # 箱ひげ図の描画
    ax1.boxplot(r_mean, positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='cyan', color='cyan'), medianprops=dict(color='blue'))
    ax1.set_ylabel('Reward', color='cyan')
    ax1.tick_params(axis='y', labelcolor='cyan')

    # P&Lのための新しいy軸
    ax2 = ax1.twinx()
    ax2.boxplot(pnl_mean, positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='magenta', color='magenta'), medianprops=dict(color='red'))
    ax2.boxplot(pnl_last, positions=[3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='yellow', color='yellow'), medianprops=dict(color='orange'))
    ax2.set_ylabel('P&L', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')

    plt.title('Boxplot of Reward, P&L, and Last P&L')
    plt.xticks([1, 2, 3], ['Mean Reward', 'Mean P&L', 'Last P&L'])
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path / "results.png")
        plt.close()
    else:
        plt.show()

