import numpy as np
from stable_baselines3.common.env_checker import check_env

from src.crypto_env import CryptoEnv
from src.crypto_agent import CryptoAgent

class DummyPolicy:
    def predict(self, observation):
        return np.random.randint(0, 4), {}

def main():
    env=CryptoEnv(
        T=50,
        alpha_realized=1,
        alpha_unrealized=1,
        agent=CryptoAgent(),
        datapath="data/test.csv"
    )

    check_env(env)

    policy=DummyPolicy()
    test_env(env, policy)

def test_env(env:CryptoEnv, policy:DummyPolicy):
    obs, _info=env.reset()
    done=False
    obs_history=[]
    reward_history=[]
    action_history=[]
    realized_pnl_rate_history=[]
    while not done:
        action, _states=policy.predict(obs)
        action_history.append(
            env.agent.actual_action(action).value
            # action
            )
        obs, reward, done, _truncated, info=env.step(action)

        obs_history.append(obs[:5])
        reward_history.append(reward)
        realized_pnl_rate_history.append(env.agent.realized_pnl_rate)

    visualize(obs_history, action_history, reward_history, realized_pnl_rate_history)

import matplotlib.pyplot as plt

def visualize(obs_history, action_history, reward_history, realized_pnl_rate_history):
    # Convert obs_history to a NumPy array for easier indexing
    obs_array = np.array(obs_history)
    reward_array = np.array(reward_history)
    realized_pnl_rate_array = np.array(realized_pnl_rate_history)

    # Plot OHLCV data
    fig, axs=plt.subplots(4, 1, figsize=(6,8))

    # Plot Open, High, Low, Close
    axs[0].plot(obs_array[:, 0], label='Open', color='blue')
    axs[0].plot(obs_array[:, 1], label='High', color='green')
    axs[0].plot(obs_array[:, 2], label='Low', color='red')
    axs[0].plot(obs_array[:, 3], label='Close', color='orange')
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
        axs[2].plot(idx, i_action*np.ones(len(idx)), markers[i_action], label=labels[i_action], color=colors[i_action])
    axs[2].set_title('Action History')
    axs[2].set_ylabel('Action Index')
    axs[2].set_yticks(ticks=[0, 1, 2, 3], labels=['Long', 'Short', 'Close', 'Stay'])
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
    plt.show()


if __name__=="__main__":
    main()
