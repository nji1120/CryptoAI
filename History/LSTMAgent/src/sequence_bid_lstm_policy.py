"""
LSTMで特徴抽出するpolicy
"""

from typing import Callable, Tuple
from dataclasses import dataclass
from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


@dataclass
class DefaultHyperParams:
    """
    Policyのハイパーパラメータのデフォルト値
    """
    ohlcv_size:int=5
    lstm_hidden_dim:int=8 # 全htをぶち込むのでhiddenは小さめに
    lstm_num_layers:int=1
    actor_dim:int=64
    actor_num_layers:int=1
    critic_dim:int=64
    critic_num_layers:int=1


class SeqBidLSTMNetwork(nn.Module):
    """
    LSTMで特徴抽出するネットワーク
    Bidirectional LSTMを使う & 全htをactorとcriticにぶち込む
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param actor_dim: (int) number of units for the last layer of the policy network
    :param critic_dim: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        lstm_hidden_dim: int = DefaultHyperParams.lstm_hidden_dim, # LSTMの隠れ層のユニット数
        lstm_num_layers: int = DefaultHyperParams.lstm_num_layers, # LSTMの層数
        actor_dim: int = DefaultHyperParams.actor_dim,
        actor_num_layers: int = DefaultHyperParams.actor_num_layers,
        critic_dim: int = DefaultHyperParams.critic_dim,
        critic_num_layers: int = DefaultHyperParams.critic_num_layers,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = actor_dim
        self.latent_dim_vf = critic_dim

        # LSTM
        lstm_input_dim=DefaultHyperParams.ohlcv_size
        sequence_length=int(feature_dim/lstm_input_dim) # 時系列データの長さ
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, batch_first=True, num_layers=lstm_num_layers, bidirectional=True)

        # Policy network
        policy_layers=[]
        for i in range(actor_num_layers):
            in_dim=lstm_hidden_dim*2*sequence_length if i==0 else actor_dim #bidirectionalなので2倍 & 時系列データの長さ分
            policy_layers+=[
                nn.Linear(in_dim, actor_dim),
                nn.Tanh()
            ]
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network
        value_layers=[]
        for i in range(critic_num_layers):
            in_dim=lstm_hidden_dim*2*sequence_length if i==0 else critic_dim #bidirectionalなので2倍 & 時系列データの長さ分
            value_layers+=[
                nn.Linear(in_dim, critic_dim),
                nn.Tanh()
            ]
        self.value_net = nn.Sequential(*value_layers)

        def log_model_summary():
            bar_num=25
            header_msg="*"*bar_num+" model summary " +"*"*bar_num
            print("\033[32m"+header_msg+"\033[0m")
            print(f"feature_extractor: {self.lstm}")
            print(f"policy_net: {self.policy_net}")
            print(f"value_net: {self.value_net}")

            print("\033[32m"+"*"*len(header_msg)+"\033[0m")
        log_model_summary()


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        h_seq=self.__get_lstm_h_seq(features)
        return self.policy_net(h_seq)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        h_seq=self.__get_lstm_h_seq(features)
        return self.value_net(h_seq)
    
    def __get_lstm_h_seq(self, features: th.Tensor) -> th.Tensor:
        """
        LSTMに通して全htを取得する関数
        """
        batch_size=features.shape[0]
        features=features.reshape(batch_size, -1, DefaultHyperParams.ohlcv_size) # [batch, seq x OHLCV_SIZE] -> [batch, seq, OHLCV_SIZE]
        h_seq, _= self.lstm(features)
        h_seq=h_seq.reshape(batch_size, -1) # [batch, seq, hidden] -> [batch, seq*hidden]

        def log_var_shape():
            print("-"*100)
            print("features.shape:",features.shape)
            print("h_seq.shape:",h_seq.shape)
            print("-"*100)
        # log_var_shape()

        return h_seq


class SeqBidLSTMNetworkPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False


        #networkのパラメータを持った辞書を記憶
        self.network_kwargs:dict=kwargs.pop("network_kwargs",{}) 


        # user_sde, ortho_init以外のキーを捨てないとエラー吐く
        kwargs_keys=list(kwargs.keys())
        for key in kwargs_keys:
            if not key in ["user_sde", "ortho_init"]:
                kwargs.pop(key)


        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:

        lstm_hidden_dim=self.network_kwargs.get("lstm_hidden_dim",DefaultHyperParams.lstm_hidden_dim)
        lstm_num_layers=self.network_kwargs.get("lstm_num_layers",DefaultHyperParams.lstm_num_layers)
        actor_dim=self.network_kwargs.get("actor_dim",DefaultHyperParams.actor_dim)
        actor_num_layers=self.network_kwargs.get("actor_num_layers",DefaultHyperParams.actor_num_layers)
        critic_dim=self.network_kwargs.get("critic_dim",DefaultHyperParams.critic_dim)
        critic_num_layers=self.network_kwargs.get("critic_num_layers",DefaultHyperParams.critic_num_layers)

        self.mlp_extractor = SeqBidLSTMNetwork(
            self.features_dim, 
            lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers, 
            actor_dim=actor_dim, actor_num_layers=actor_num_layers,
            critic_dim=critic_dim, critic_num_layers=critic_num_layers
        )


