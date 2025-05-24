"""
Transformerで特徴抽出するpolicy
"""

from typing import Callable, Tuple
from dataclasses import dataclass
from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

from .utils import create_positional_encoding, DefaultHyperParams


class SequenceTransformerNetwork(nn.Module):
    """
    Transformerで特徴抽出するネットワーク  
    actorとcriticに全シーケンスをぶちこむ  
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param actor_dim: (int) number of units for the last layer of the policy network
    :param critic_dim: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        transformer_hidden_dim: int = DefaultHyperParams.transformer_hidden_dim,
        transformer_n_heads: int = DefaultHyperParams.transformer_n_heads,
        transformer_n_layers: int = DefaultHyperParams.transformer_n_layers,
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

        sequence_length=int(feature_dim//DefaultHyperParams.ohlcv_size) # シーケンス長さ
        self.positional_encoding:th.Tensor=create_positional_encoding(sequence_length, transformer_hidden_dim)

        # 入力をtransoformerの入力次元に変換するconv1d
        self.input2hidden_layer=nn.Conv1d(
            DefaultHyperParams.ohlcv_size,
            transformer_hidden_dim,
            kernel_size=1
        )

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                transformer_hidden_dim, transformer_n_heads,
                batch_first=True
            ),
            num_layers=transformer_n_layers
        )


        # Policy network
        policy_layers=[]
        for i in range(actor_num_layers):
            in_dim=transformer_hidden_dim * sequence_length if i==0 else actor_dim
            policy_layers+=[
                nn.Linear(in_dim, actor_dim),
                nn.Tanh()
            ]
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network
        value_layers=[]
        for i in range(critic_num_layers):
            in_dim=transformer_hidden_dim * sequence_length if i==0 else critic_dim
            value_layers+=[
                nn.Linear(in_dim, critic_dim),
                nn.Tanh()
            ]
        self.value_net = nn.Sequential(*value_layers)


        def log_model_summary():
            bar_num=25
            header_msg="*"*bar_num+" model summary " +"*"*bar_num
            print("\033[32m"+header_msg+"\033[0m")
            print(f"feature_extractor: {self.transformer}")
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
        transformer_out=self.__get_transformer_out(features)
        return self.policy_net(transformer_out)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        transformer_out=self.__get_transformer_out(features)
        return self.value_net(transformer_out)
    
    def __get_transformer_out(self, features: th.Tensor) -> th.Tensor:
        """
        Transformerに通して全てのシーケンス分の出力を取得する関数
        """
        device=features.device

        batch_size=features.shape[0]
        features=features.reshape(batch_size, -1, DefaultHyperParams.ohlcv_size) # [batch, seq x OHLCV_SIZE] -> [batch, seq, OHLCV_SIZE]
        
        # 入力をtransoformerの入力次元に変換するconv1d
        features=features.transpose(1, 2) # [batch, seq, OHLCV_SIZE] -> [batch, OHLCV_SIZE, seq]
        transformer_in: th.Tensor = self.input2hidden_layer(features)

        # transformerに通す
        transformer_in=transformer_in.transpose(1, 2) # [batch, transformer_hidden_dim, seq] -> [batch, seq, transformer_hidden_dim]
        transformer_in=transformer_in+self.positional_encoding.to(device) # 位置エンコーディングを加算
        transformer_out: th.Tensor = self.transformer(transformer_in) # [batch, seq, transformer_hidden_dim]
        
        def log_var_shape():
            print("-"*100)
            print("features.shape:",features.shape)
            print("transformer_in.shape:",transformer_in.shape)
            print("transformer_out.shape:",transformer_out.shape)
            print("-"*100)
        # log_var_shape()

        return transformer_out.reshape(batch_size, -1) # [batch, seq, transformer_hidden_dim] -> [batch, seq * transformer_hidden_dim]


class SequenceTransformerNetworkPolicy(ActorCriticPolicy):
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

        transformer_hidden_dim=self.network_kwargs.get("transformer_hidden_dim",DefaultHyperParams.transformer_hidden_dim)
        transformer_n_heads=self.network_kwargs.get("transformer_n_heads",DefaultHyperParams.transformer_n_heads)
        transformer_n_layers=self.network_kwargs.get("transformer_n_layers",DefaultHyperParams.transformer_n_layers)
        actor_dim=self.network_kwargs.get("actor_dim",DefaultHyperParams.actor_dim)
        actor_num_layers=self.network_kwargs.get("actor_num_layers",DefaultHyperParams.actor_num_layers)
        critic_dim=self.network_kwargs.get("critic_dim",DefaultHyperParams.critic_dim)
        critic_num_layers=self.network_kwargs.get("critic_num_layers",DefaultHyperParams.critic_num_layers)

        self.mlp_extractor = SequenceTransformerNetwork(
            self.features_dim, 
            transformer_hidden_dim=transformer_hidden_dim, transformer_n_heads=transformer_n_heads, transformer_n_layers=transformer_n_layers, 
            actor_dim=actor_dim, actor_num_layers=actor_num_layers,
            critic_dim=critic_dim, critic_num_layers=critic_num_layers
        )


