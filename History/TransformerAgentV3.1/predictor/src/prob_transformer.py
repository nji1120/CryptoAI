"""
Transformerで値を予測するモデル
出力は確率分布のPNN形式
"""

import torch as th
from torch import nn

from .utils import create_positional_encoding


class ProbTransformer(nn.Module):
    """
    Transformerで値を予測するモデル.
    予測は確率分布のPNN形式とする. (信頼度がほしいため)
    """
    @classmethod
    def from_config(cls, config: dict):
        """
        configを渡すと, インスタンスを自動生成してくれる.
        config: dict
            - model.input_params
            - model.transformer
            - model.out_layer
        """
        # 必要なパラメータをconfigから取得
        sequence_length = config["transformer"]["sequence_length"]
        input_feature_dim = config["transformer"]["feature_dim"]
        future_time_dim = config["transformer"]["future_time_dim"]
        transformer_feedforward_dim = config["transformer"]["feedforward_dim"]
        transformer_hidden_dim = config["transformer"]["hidden_dim"]
        transformer_n_heads = config["transformer"]["n_heads"]
        transformer_n_layers = config["transformer"]["n_layers"]
        transformer_dropout = config["transformer"]["dropout"]
        output_n_layers = config["out_layer"]["n_layers"]
        output_hidden_dim = config["out_layer"]["hidden_dim"]
        output_dim = config["out_layer"]["out_dim"]

        return cls(
            sequence_length=sequence_length,
            input_feature_dim=input_feature_dim,
            future_time_dim=future_time_dim,
            transformer_feedforward_dim=transformer_feedforward_dim,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_n_heads=transformer_n_heads,
            transformer_n_layers=transformer_n_layers,
            transformer_dropout=transformer_dropout,
            output_n_layers=output_n_layers,
            output_hidden_dim=output_hidden_dim,
            output_dim=output_dim,
        )
    
    def __init__(
        self,
        sequence_length: int,
        input_feature_dim: int,
        future_time_dim: int,
        transformer_feedforward_dim: int = 256,
        transformer_hidden_dim: int = 128,
        transformer_n_heads: int = 8,
        transformer_n_layers: int = 4,
        transformer_dropout: float = 0.3,
        output_n_layers: int = 2,
        output_hidden_dim: int = 64,
        output_dim: int=5,
    ):
        """
        :param sequence_length: 過去の時系列データの長さ
        :param input_feature_dim: 入力の次元
        :param future_time_dim: 予測する未来の時刻の次元
        """
        super().__init__()

        self.future_time_dim=future_time_dim
        self.output_dim=output_dim # policyの方で使うために残しておく

        cls_token_length=1
        self.positional_encoding:th.Tensor=create_positional_encoding(
            sequence_length+cls_token_length, transformer_hidden_dim
        ) #clsトークン分の1を加算

        # 入力をtransoformerの入力次元に変換するconv1d
        self.input2hidden_layer=nn.Conv1d(
            input_feature_dim,
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
                batch_first=True,
                dim_feedforward=transformer_feedforward_dim,
                dropout=transformer_dropout
            ),
            num_layers=transformer_n_layers
        )
        
        # 出力層 : clsトークンの特徴量を受け取る. 平均とlog分散を出力する
        self.mu_layer=self.__create_output_layer(
            transformer_hidden_dim+self.future_time_dim, 
            output_n_layers, output_hidden_dim, output_dim
        )
        self.logvar_layer=self.__create_output_layer(
            transformer_hidden_dim+self.future_time_dim, 
            output_n_layers, output_hidden_dim, output_dim
        )

    
    def __create_output_layer(self, input_dim: int, output_n_layers: int, output_hidden_dim: int, output_dim: int):
        output_layers=[]
        for i in range(output_n_layers):
            in_dim=input_dim if i==0 else output_hidden_dim
            output_layers+=[
                nn.Linear(in_dim, output_hidden_dim),
                nn.LeakyReLU()
            ]
        output_layers+=[
            nn.Linear(output_hidden_dim, output_dim),
        ]
        return nn.Sequential(*output_layers)


    def __time2onehot(self, time: th.Tensor) -> th.Tensor:
        """
        :param time: [batch]
        :return: [batch x time_dim]
        """
        onehot=th.zeros(time.shape[0], self.future_time_dim)
        for i in range(self.future_time_dim):
            onehot[:, i]=th.where(time==i, 1, 0)
        return onehot.to(th.float32)


    def forward(self, x: th.Tensor, time: th.Tensor) -> th.Tensor:
        """
        :param x: [batch x sequence x feature_dim]
        :param time: [batch]
        :return: out_mu: [batch x output_dim], out_logvar: [batch x output_dim]
        """

        # 入力をtransformerの入力次元に合わせる (埋め込みに近い)
        x=x.transpose(1, 2) # [batch x sequence x feature_dim] -> [batch x feature_dim x sequence]
        x=self.input2hidden_layer(x) # [batch x feature_dim x sequence] -> [batch x transformer_hidden_dim x sequence]
        x=x.transpose(1, 2) # [batch x transformer_hidden_dim x sequence] -> [batch x sequence x transformer_hidden_dim]

        # CLSトークンを追加
        cls_token=self.cls_token.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1, 1) # [batch x transformer_hidden_dim x 1]
        x=th.cat([cls_token, x], dim=1) # [batch x (sequence+1) x transformer_hidden_dim]

        # positional encodingを加算
        x=x+self.positional_encoding.to(x.device)

        # transformerに通す
        h=self.transformer(x)

        # clsトークンの特徴量を取得
        h_cls=h[:, 0, :] # [batch x transformer_hidden_dim]

        # 予測する未来の時刻をone-hotに変換
        time_onehot=self.__time2onehot(time) # [batch x future_time_dim]

        # 予測する未来の時刻をconcat
        # print(f"h_cls.shape: {h_cls.shape}, time_onehot.shape: {time_onehot.shape}")
        h_cls=th.cat([h_cls, time_onehot], dim=1) # [batch x (transformer_hidden_dim+future_time_dim)]

        # 出力層に通す
        mu=self.mu_layer(h_cls) # [batch x output_dim]
        logvar=self.logvar_layer(h_cls) # [batch x output_dim]

        return mu, logvar
        
        
        
    def predict(self, x: th.Tensor, time: th.Tensor) -> th.Tensor:
        """
        :param x: [batch x sequence x feature_dim]
        :param time: [batch]
        :return: mu: [batch x output_dim], logvar: [batch x output_dim]
        """
        with th.no_grad():  
            mu, logvar=self.forward(x, time)
        return mu, logvar
        