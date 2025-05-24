import torch as th
from dataclasses import dataclass
def create_positional_encoding(seq_len: int, d_model: int) -> th.Tensor:
    """
    Positional encodingを生成する関数
    """
    position = th.arange(seq_len, dtype=th.float).unsqueeze(1)
    div_term = th.exp(th.arange(0, d_model, 2).float() * -(th.log(th.tensor(10000.0)) / d_model))
    pe = th.zeros(seq_len, d_model)
    pe[:, 0::2] = th.sin(position * div_term)
    pe[:, 1::2] = th.cos(position * div_term)
    return pe.unsqueeze(0)  # バッチ次元を追加


@dataclass
class DefaultHyperParams:
    """
    Policyのハイパーパラメータのデフォルト値
    """
    ohlcv_size:int=5
    transformer_hidden_dim:int=512
    transformer_n_heads:int=8
    transformer_n_layers:int=6
    actor_dim:int=256
    actor_num_layers:int=2
    critic_dim:int=256
    critic_num_layers:int=2
