import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor

from src.crypto_env import CryptoEnv


def get_action_prob(policy:PPO, obs:np.ndarray):
    """
    policyのaction_probを返す
    """
    with torch.no_grad():
        obs_tr=torch.Tensor(obs).unsqueeze(0).to(policy.policy.device)
        distribute=policy.policy.get_distribution(obs_tr)
        probs_tr=distribute.distribution.probs
        probs=probs_tr.detach().cpu().numpy()
    return probs


def test_env(env:CryptoEnv, policy:PPO, n_test:int=1, save_path:Path=None):
    
    results=[]

    print(f"\033[32m--- START test_env "+"-"*50+"\033[0m")
    for i_test in tqdm(range(n_test), desc="test_env"):

        obs_history=[]
        reward_history=[]
        action_history=[]
        action_prob_history=[]
        realized_pnl_rate_history=[] # 累積確定利益率

        obs,_=env.reset()
        done=False
        while not done:
            action, _states=policy.predict(obs, deterministic=False)
            action_history.append(
                action
            )

            action_prob=get_action_prob(policy, obs)[0]
            action_prob_history.append(action_prob)

            current_price=env.get_current_data() #現stepのprice (このopenのみactorに渡されている)
            future_price=env.original_df.iloc[env.current_idx+env.s_future]["close"] # 未来のcloseデータ
            obs, reward, done,_,info=env.step(action)

            obs_history.append(
                np.concatenate([current_price.values.flatten(), [future_price]])
            )
            reward_history.append(reward)
            realized_pnl_rate_history.append(
                env.agent.cumlative_realized_pnl_rate
            )

        # if env.agent.cumlative_realized_pnl_rate>1:
        #     print(f"\n!!!!! i:{i_test} PnL% {env.agent.cumlative_realized_pnl_rate} !!!!!")
        #     print(obs)

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
            obs_history, action_history, action_prob_history,
            reward_history, realized_pnl_rate_history, 
            save_path/"test_imgs", 
            filename=f"result_{i_test}.png"
        )

    results_df=pd.DataFrame(results)
    results_df.to_csv(save_path/"results.csv", index=False)
    visualize_results(results_df, save_path)

    print(f"\033[32m--- END test_env --"+"-"*50+"\033[0m")


def visualize_history(
        obs_history, action_history, action_prob_history,
        reward_history, realized_pnl_rate_history, 
        save_path:Path=None, filename:str="test.png"
    ):
    # Convert obs_history to a NumPy array for easier indexing
    obs_array = np.array(obs_history)
    reward_array = np.array(reward_history)
    realized_pnl_rate_array = np.array(realized_pnl_rate_history)
    action_prob_array = np.array(action_prob_history)

    # Plot OHLCV data
    axs:list[plt.Axes]=[]
    fig, axs=plt.subplots(4, 1, figsize=(6,8))

    # Plot Open, High, Low, Close
    axs[0].plot(obs_array[:, 0], label='Open', color='green', alpha=1)
    axs[0].plot(obs_array[:, 1], label='High', color='blue', alpha=0.2)
    axs[0].plot(obs_array[:, 2], label='Low', color='red', alpha=0.2)
    axs[0].plot(obs_array[:, 3], label='Close', color='orange', alpha=1)
    axs[0].plot(obs_array[:, 5], label='Future', color='purple', alpha=1, linestyle='--')
    axs[0].set_title('OHLC Data')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # Plot Volume
    axs[1].plot(obs_array[:, 4], label='Volume', color='purple')
    axs[1].set_title('Volume')
    axs[1].set_ylabel('Volume')
    axs[1].legend()

    # Plot Actions openの位置にpositionをプロット
    markers=["o", "x", "s", "D"]
    colors=["red", "blue", "green", "gray"]
    labels=["Long", "Short", "Close", "Stay"]
    for i_action in range(4):
        idx=np.where(np.array(action_history)==i_action)[0]
        axs[0].plot(
            idx, obs_array[idx, 0], markers[i_action], 
            label=labels[i_action], color=colors[i_action],
            markersize=4
        )  # Plot on Close


    # Plot action_prob as stacked bar chart
    axs[2].set_title('Action Probability')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('Probability')

    # LongとShortの確率をスタックして描画
    long_probs = action_prob_array[:, 0]  # Longの確率
    short_probs = action_prob_array[:, 1]  # Shortの確率
    axs[2].bar(range(len(long_probs)), long_probs, label='Long', color='red', alpha=0.5)
    axs[2].bar(range(len(short_probs)), short_probs, bottom=long_probs, label='Short', color='blue', alpha=0.5)
    axs[2].hlines(0.75, 0, len(long_probs), color='gray', linestyle='--', linewidth=1)
    axs[2].hlines(0.25, 0, len(long_probs), color='gray', linestyle='--', linewidth=1)

    axs[2].legend()

    # Plot Immediate Rewards
    axs[3].set_xlabel('Time Steps')
    axs[3].set_ylabel('Immediate Reward', color='cyan')
    axs[3].plot(reward_array, label='Immediate Reward', color='cyan')
    axs[3].tick_params(axis='y', labelcolor='cyan')

    # Create a second y-axis for Cumulative Reward
    ax2 = axs[3].twinx()
    ax2.set_ylabel('Realized PNL Rate', color='magenta')
    ax2.plot(realized_pnl_rate_array, label='Realized PNL Rate', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')

    axs[3].set_title('Rewards / Realized PNL Rate')

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

