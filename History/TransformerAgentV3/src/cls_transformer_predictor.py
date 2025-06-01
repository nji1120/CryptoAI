"""
Transformerで値を予測するモデル
出力は確率分布のPNN形式
"""

import torch as th
from torch import nn

from .utils import create_positional_encoding, DefaultHyperParams


class ClsTransformerPredictor(nn.Module):
    """
    Transformerで値を予測するモデル.
    予測は確率分布のPNN形式とする. (信頼度がほしいため)
    """
    def __init__(
        self,
        sequence_length: int,
        transformer_feed_forward_dim: int = DefaultHyperParams.transformer_feed_forward_dim,
        transformer_hidden_dim: int = DefaultHyperParams.transformer_hidden_dim,
        transformer_n_heads: int = DefaultHyperParams.transformer_n_heads,
        transformer_n_layers: int = DefaultHyperParams.transformer_n_layers,
        output_n_layers: int = 2,
        output_hidden_dim: int = 256,
        output_dim: int = 2,
    ):
        super().__init__()

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
                batch_first=True,
                dim_feedforward=transformer_feed_forward_dim
            ),
            num_layers=transformer_n_layers
        )
        
        # 出力層 : clsトークンの特徴量を受け取る
        output_layers=[]
        for i in range(output_n_layers):
            in_dim=transformer_hidden_dim if i==0 else output_hidden_dim
            output_layers+=[
                nn.Linear(in_dim, output_hidden_dim),
                nn.ReLU()
            ]
        output_layers+=[
            nn.Linear(output_hidden_dim, output_dim),
            nn.ReLU()
        ]
        self.output_layer=nn.Sequential(*output_layers)



    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        :param x: [batch x sequence x feature_dim]
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

        # 出力層に通す
        out=self.output_layer(h_cls) # [batch x output_dim]

        return out
        
        
        
        
        