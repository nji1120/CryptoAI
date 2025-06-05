"""
Transformerで特徴抽出するpolicy
"""

from typing import Callable, Tuple
from dataclasses import dataclass
from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

from .utils import create_positional_encoding

from pathlib import Path
MODEL_ROOT=Path(__file__).parent.parent
import sys
sys.path.append(str(MODEL_ROOT.parent))
from predictor.src.prob_transformer import ProbTransformer


class ClsTransformerNetwork(nn.Module):
    """
    Transformerで特徴抽出を行うネットワーク  
    actorとcriticにはCLSトークンの出力（系列全体の集約ベクトル）のみを入力する  
    :param flattened_feature_dim: 1次元化したときの入力の次元数 [timestep x feature_dim]
    :param actor_dim: ポリシーネットワークの最終層のユニット数
    :param critic_dim: 価値ネットワークの最終層のユニット数
    """    
    def __init__(
        self,
        feature_dim: int,
        flattened_feature_dim: int,
        transformer_feed_forward_dim: int,
        transformer_hidden_dim: int,
        transformer_n_heads: int,
        transformer_n_layers: int,
        transformer_dropout: float,
        actor_dim: int,
        actor_num_layers: int,
        critic_dim: int,
        critic_num_layers: int,
        predictor: ProbTransformer, # 学習済みの予測器
        s_future: int, # 予測器の予測時間
    ):
        super().__init__()

        self.feature_dim=feature_dim

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = actor_dim
        self.latent_dim_vf = critic_dim

        sequence_length=int(flattened_feature_dim//feature_dim) # シーケンス長さ
        cls_token_length=1
        self.positional_encoding:th.Tensor=create_positional_encoding(
            cls_token_length+sequence_length+s_future, transformer_hidden_dim
        ) #clsトークン分の1を加算

        # 入力をtransoformerの入力次元に変換するconv1d
        self.input2hidden_layer=nn.Conv1d(
            feature_dim,
            transformer_hidden_dim,
            kernel_size=1,
        )


        # **************** 予測器とその出力をtransformerの入力次元に変換するconv1d ****************
        self.predictor=predictor
        self.s_future=s_future
        self.predict2hidden_layer=nn.Conv1d(
            predictor.output_dim*2, # muとlogavarのconcat分
            transformer_hidden_dim,
            kernel_size=1,
        )
        # **************************************************************************************


        # CLSトークン
        self.cls_token_dim=transformer_hidden_dim
        self.cls_token=nn.Parameter(th.randn(self.cls_token_dim)) #CLSトークンは学習するらしい

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                transformer_hidden_dim, transformer_n_heads,
                batch_first=True,
                dim_feedforward=transformer_feed_forward_dim,
                dropout=transformer_dropout
            ),
            num_layers=transformer_n_layers
        )


        # Policy network
        policy_layers=[]
        for i in range(actor_num_layers):
            in_dim=transformer_hidden_dim if i==0 else actor_dim
            policy_layers+=[
                nn.Linear(in_dim, actor_dim),
                nn.LeakyReLU()
            ]
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network
        value_layers=[]
        for i in range(critic_num_layers):
            in_dim=transformer_hidden_dim if i==0 else critic_dim
            value_layers+=[
                nn.Linear(in_dim, critic_dim),
                nn.LeakyReLU()
            ]
        self.value_net = nn.Sequential(*value_layers)


        def log_model_summary():
            bar_num=50
            header_msg="*"*bar_num+" model summary " +"*"*bar_num
            print("\033[32m"+header_msg+"\033[0m")
            print(f"feature_extractor: {self.transformer}")
            print(f"cls_token shape: {self.cls_token.shape}")
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
        features=features.reshape(batch_size, -1, self.feature_dim) # [batch, seq x OHLCV_SIZE] -> [batch, seq, OHLCV_SIZE]

        # >>>>>>>>>>>>>>>>>> predictorによる予測 >>>>>>>>>>>>>>>>>>>>>>>>
        h_predictor=self.__get_predictor_feature(features)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        # 入力をtransoformerの入力次元に変換するconv1d
        features=features.transpose(1, 2) # [batch, seq, OHLCV_SIZE] -> [batch, OHLCV_SIZE, seq]
        transformer_in: th.Tensor = self.input2hidden_layer(features)
        transformer_in=transformer_in.transpose(1, 2) # [batch, transformer_hidden_dim, seq] -> [batch, seq, transformer_hidden_dim]

        # transformerに通す
        transformer_in=th.cat([
            self.cls_token.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1),
            transformer_in,
            h_predictor
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
    

    def __get_predictor_feature(self, x: th.Tensor) -> th.Tensor:
        """
        predictorで予測し, transformerの入力次元に変換する
        :param x: [batch, seq, feature_dim]
        :return: [batch, seq, transformer_hidden_dim]
        """

        device=x.device
        batch_size=x.shape[0]
        ts=th.arange(1, self.s_future+1).to(device)
        

        # # >> バッチ1つ1つに対して予測 (ここが学習時間のボトルネックになりそう...) >>
        # out_predictor=th.zeros(batch_size, self.s_future, self.predictor.output_dim*2).to(device)
        # for i_batch in range(batch_size):
        #     input_data=x[i_batch].unsqueeze(0).repeat(self.s_future, 1, 1) # 未来分複製する
        #     mu, logvar=self.predictor.predict(input_data, ts) # [s_future, output_dim]
        #     out_predictor[i_batch]=th.cat([mu, logvar], dim=1) # [s_future, output_dim*2]
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        # >> 一気に予測する実装 (メモリは喰うがこっちのほうが速いはず) >>
        ts_expand = ts.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
        x_expand = x.unsqueeze(1).repeat(1, self.s_future, 1, 1).reshape(-1, x.shape[1], x.shape[2])
        mu, logvar = self.predictor.predict(x_expand, ts_expand)
        mu = mu.view(batch_size, self.s_future, -1)
        logvar = logvar.view(batch_size, self.s_future, -1)
        out_predictor = th.cat([mu, logvar], dim=2)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        out_predictor=out_predictor.transpose(1, 2) # [batch, s_future, output_dim*2] -> [batch, output_dim*2, s_future]
        h_predictor:th.Tensor=self.predict2hidden_layer(out_predictor)
        h_predictor=h_predictor.transpose(1, 2) # [batch, hidden_dim, s_future] -> [batch, s_future, hidden_dim]


        return h_predictor
        


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

        # モデルのパラメータを取得
        input_args:dict=self.network_kwargs.get("input_params",{})
        transformer_args:dict=self.network_kwargs.get("transformer",{})
        out_layer_args:dict=self.network_kwargs.get("out_layer",{})

        transformer_hidden_dim=transformer_args.get("hidden_dim",128)
        transformer_feed_forward_dim=transformer_args.get("feedforward_dim",256)
        transformer_n_heads=transformer_args.get("n_heads",8)
        transformer_n_layers=transformer_args.get("n_layers",4)
        transformer_dropout=transformer_args.get("dropout",0.3)

        actor_dim=out_layer_args.get("hidden_dim",64)
        actor_num_layers=out_layer_args.get("n_layers",2)
        critic_dim=out_layer_args.get("hidden_dim",64)
        critic_num_layers=out_layer_args.get("n_layers",2)

        predictor=self.network_kwargs.get("predictor",None)
        s_future=self.network_kwargs.get("s_future",1)

        self.mlp_extractor = ClsTransformerNetwork(

            feature_dim=input_args.get("feature_dim",5), 
            flattened_feature_dim=self.features_dim, 
            transformer_feed_forward_dim=transformer_feed_forward_dim, transformer_hidden_dim=transformer_hidden_dim, 
            transformer_n_heads=transformer_n_heads, transformer_n_layers=transformer_n_layers, 
            transformer_dropout=transformer_dropout,
            actor_dim=actor_dim, actor_num_layers=actor_num_layers,
            critic_dim=critic_dim, critic_num_layers=critic_num_layers,
            predictor=predictor,
            s_future=s_future
        )


