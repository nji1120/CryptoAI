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


class ClsTransformerNetwork(nn.Module):
    """
    Transformerで特徴抽出を行うネットワーク  
    actorとcriticにはCLSトークンの出力（系列全体の集約ベクトル）のみを入力する  
    :param feature_dim: 特徴抽出器（例: CNN）で得られる特徴量の次元数
    :param actor_dim: ポリシーネットワークの最終層のユニット数
    :param critic_dim: 価値ネットワークの最終層のユニット数
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

        sequence_length=int(feature_dim//DefaultHyperParams.in_feature_dim) # シーケンス長さ
        cls_token_length=1
        self.positional_encoding:th.Tensor=create_positional_encoding(
            sequence_length+cls_token_length, transformer_hidden_dim
        ) #clsトークン分の1を加算

        # 入力をtransoformerの入力次元に変換するconv1d
        self.input2hidden_layer=nn.Conv1d(
            DefaultHyperParams.in_feature_dim,
            transformer_hidden_dim,
            kernel_size=1,
        )

        # CLSトークン
        self.cls_token_dim=transformer_hidden_dim
        self.cls_token=nn.Parameter(th.randn(self.cls_token_dim)) #CLSトークンは学習するらしい

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
            in_dim=transformer_hidden_dim if i==0 else actor_dim
            policy_layers+=[
                nn.Linear(in_dim, actor_dim),
                nn.ReLU()
            ]
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network
        value_layers=[]
        for i in range(critic_num_layers):
            in_dim=transformer_hidden_dim if i==0 else critic_dim
            value_layers+=[
                nn.Linear(in_dim, critic_dim),
                nn.ReLU()
            ]
        self.value_net = nn.Sequential(*value_layers)


        def log_model_summary():
            bar_num=50
            header_msg="*"*bar_num+" model summary " +"*"*bar_num
            print("\033[32m"+header_msg+"\033[0m")
            print(f"feature_extractor: {self.transformer}")
            print(f"cls_token: {self.cls_token}")
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
        transformer_out=self.__get_cls_feature(features)
        return self.policy_net(transformer_out)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        transformer_out=self.__get_cls_feature(features)
        return self.value_net(transformer_out)
    
    def __get_cls_feature(self, features: th.Tensor) -> th.Tensor:
        """
        Transformerに通してCLSトークンの特徴量を取得する関数
        """
        device=features.device

        batch_size=features.shape[0]
        features=features.reshape(batch_size, -1, DefaultHyperParams.in_feature_dim) # [batch, seq x OHLCV_SIZE] -> [batch, seq, OHLCV_SIZE]
        
        # 入力をtransoformerの入力次元に変換するconv1d
        features=features.transpose(1, 2) # [batch, seq, OHLCV_SIZE] -> [batch, OHLCV_SIZE, seq]
        transformer_in: th.Tensor = self.input2hidden_layer(features)

        # transformerに通す
        transformer_in=transformer_in.transpose(1, 2) # [batch, transformer_hidden_dim, seq] -> [batch, seq, transformer_hidden_dim]
        transformer_in=th.cat([
            self.cls_token.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
            transformer_in
        ], dim=1)
        transformer_in=transformer_in+self.positional_encoding.to(device) # 位置エンコーディングを加算
        transformer_out: th.Tensor = self.transformer(transformer_in) # [batch, seq, transformer_hidden_dim]

        cls_feature=transformer_out[:, 0, :] # CLSトークンの特徴量
        
        def log_var_shape():
            print("-"*100)
            print("features.shape:",features.shape)
            print("transformer_in.shape:",transformer_in.shape)
            print("transformer_out.shape:",transformer_out.shape)
            print("cls_feature.shape:",cls_feature.shape)
            print("-"*100)
        # log_var_shape()

        return cls_feature # [batch, transformer_hidden_dim]


class ClsTransformerNetworkPolicy(ActorCriticPolicy):
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

        self.mlp_extractor = ClsTransformerNetwork(
            self.features_dim, 
            transformer_hidden_dim=transformer_hidden_dim, transformer_n_heads=transformer_n_heads, transformer_n_layers=transformer_n_layers, 
            actor_dim=actor_dim, actor_num_layers=actor_num_layers,
            critic_dim=critic_dim, critic_num_layers=critic_num_layers
        )


