from pathlib import Path
PARENT_DIR=Path(__file__).parent
MODEL_ROOT=Path(__file__).parent.parent
PROJECT_ROOT=MODEL_ROOT.parent.parent
import sys
sys.path.append(str(MODEL_ROOT))

import argparse
import yaml
from yaml import safe_load as yaml_safe_load
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data_formatter import DataFormatter
from src.cls_transformer_predictor import ClsTransformerPredictor

class GaussianNLL():
    """
    PNN学習用の損失関数
    """
    def __call__(self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param mu: [batch x output_dim]
        :param logvar: [batch x output_dim]
        :param target: [batch x output_dim]
        :return: loss: [batch]
        """
        var=torch.exp(logvar)
        nll=0.5*torch.log(2*torch.pi) + torch.log(torch.sqrt(var)) + ((target-mu)**2)/(2*var)
        return torch.mean(nll)


def create_dataset(datapath: Path, data_formatter: DataFormatter, t_past: int, t_future: int) -> torch.Tensor:
    """
    データセットの作成
    """
    data=pd.read_csv(datapath)[["open", "high", "low", "close", "volume"]].values
    data_dict=data_formatter.get_formatted_data(t_past, t_future, data)
    return data_dict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=MODEL_ROOT/"test/test_train_predictor/config.yml")
    args=parser.parse_args()

    conf=yaml_safe_load(open(args.config))
    model_conf=conf["model"]
    train_conf=conf["train"]

    # **************************************** modelの準備 *************************************************
    # modelの作成
    model=ClsTransformerPredictor(
        sequence_length=model_conf["input_params"]["t_past"],
        transformer_hidden_dim=model_conf["transformer"]["hidden_dim"],
        transformer_feedforward_dim=model_conf["transformer"]["feedforward_dim"],
        transformer_n_heads=model_conf["transformer"]["n_heads"],
        transformer_n_layers=model_conf["transformer"]["n_layers"],
        output_n_layers=model_conf["out_layer"]["n_layers"],
        output_hidden_dim=model_conf["out_layer"]["hidden_dim"],
        output_dim=model_conf["out_layer"]["out_dim"],
    )
    print(model)

    # **************************************** データの準備 *************************************************
    # データ整形器の作成
    data_formatter=DataFormatter()

    # データの読み込み
    train_datapath=Path(PROJECT_ROOT/train_conf["datapath"]["train"])
    train_data_dict=create_dataset(train_datapath, data_formatter, model_conf["input_params"]["t_past"], model_conf["input_params"]["t_future"])
    
    test_datapath=Path(PROJECT_ROOT/train_conf["datapath"]["test"])
    test_data_dict=create_dataset(test_datapath, data_formatter, model_conf["input_params"]["t_past"], model_conf["input_params"]["t_future"])
    
    # **************************************** 学習の準備 *************************************************

    # 学習パラメータ
    total_timesteps=train_conf["total_timesteps"]
    save_interval=train_conf["save_interval"]
    model_name=train_conf["model_name"]
    log_name=train_conf["log_name"]
    batch_size=train_conf["batch_size"]
    learning_rate=train_conf["learning_rate"]

    # 損失関数
    criterion=GaussianNLL()

    # optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    

if __name__=="__main__":
    main()
