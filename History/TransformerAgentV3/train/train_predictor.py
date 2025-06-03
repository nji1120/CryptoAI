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
import numpy as np
import math
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from src.data_formatter import DataFormatter
from src.cls_transformer_predictor import ClsTransformerPredictor



def create_dataset(datapath: Path, data_formatter: DataFormatter, t_past: int, t_future: int) -> torch.Tensor:
    """
    データセットの作成
    """
    data=pd.read_csv(datapath)[["open", "high", "low", "close", "volume"]].values
    data_dict=data_formatter.get_formatted_data(t_past, t_future, data)
    return data_dict


def create_sequence(data:torch.Tensor,t_list:torch.Tensor,t_past:int) -> torch.Tensor:
    """
    シーケンスの作成
    :param data: [total_timesteps x feature_dim]
    :param t_list: [batch]
    :param t_past: 過去の時間ステップ数
    :return: sequence: [batch x t_past x feature_dim]
    """
    batch_size=len(t_list)
    sequence_data=torch.zeros((batch_size, t_past, data.shape[1]),device=data.device)
    for i,t in enumerate(t_list):
        sequence_data[i]=data[t-t_past+1:t+1]
    return sequence_data

def creat_dataloader(total_timesteps:int,t_past:int,t_future:int,batch_size:int,shuffle:bool,generator:torch.Generator) -> torch.utils.data.DataLoader:
    index=np.arange(t_past,total_timesteps-t_future)
    return torch.utils.data.DataLoader(index, batch_size=batch_size, shuffle=shuffle,generator=generator)


def create_result_path(root:Path, model_name:str) -> Path:
    current_time=datetime.now().strftime("%Y%m%d_%H%M%S")
    return root/model_name/current_time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=MODEL_ROOT/"test/test_train_predictor/config.yml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()

    device=args.device

    conf=yaml_safe_load(open(args.config,"r",encoding="utf-8"))
    model_conf=conf["model"]
    train_conf=conf["train"]

    result_path=create_result_path(MODEL_ROOT/"outputs/",train_conf["model_name"])
    os.makedirs(result_path,exist_ok=True)

    with open(result_path/"config.yml","w",encoding="utf-8") as f:
        raw_txt=open(args.config,"r",encoding="utf-8").readlines()
        for line in raw_txt:
            f.write(line)

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
    model.to(device)
    print(model)

    # **************************************** データの準備 *************************************************
    # データ整形器の作成
    data_formatter=DataFormatter()

    # データの読み込み
    train_datapath=Path(PROJECT_ROOT/train_conf["datapath"]["train"])
    train_data_dict=create_dataset(train_datapath, data_formatter, model_conf["input_params"]["t_past"], model_conf["input_params"]["t_future"])
    for key in train_data_dict.keys():
        train_data_dict[key]=torch.from_numpy(train_data_dict[key]).float().to(device)

    test_datapath=Path(PROJECT_ROOT/train_conf["datapath"]["test"])
    test_data_dict=create_dataset(test_datapath, data_formatter, model_conf["input_params"]["t_past"], model_conf["input_params"]["t_future"])
    for key in test_data_dict.keys():
        test_data_dict[key]=torch.from_numpy(test_data_dict[key]).float().to(device)

    # **************************************** 学習の準備 *************************************************

    # 学習パラメータ
    t_past=model_conf["input_params"]["t_past"]
    t_future=model_conf["input_params"]["t_future"]
    total_epochs=train_conf["total_epochs"]
    batch_size=train_conf["batch_size"]
    learning_rate=train_conf["learning_rate"]
    save_interval=train_conf["save_interval"]
    model_name=train_conf["model_name"]
    log_name=train_conf["log_name"]

    # 損失関数
    criterion=torch.nn.GaussianNLLLoss()

    # optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # indexの作成
    train_index_loader=creat_dataloader(
        len(train_data_dict["input"]),t_past,t_future,
        batch_size,True,torch.Generator(device=device)
    )
    test_index_loader=creat_dataloader(
        len(test_data_dict["input"]),t_past,t_future,
        batch_size,False,torch.Generator(device=device)
    )

    # tensorboardの準備
    writer=SummaryWriter(log_dir=result_path/"logs")

    # 学習
    for epoch in tqdm(range(total_epochs)):
        model.train()
        train_loss=0
        for batch_index in train_index_loader:
            input=create_sequence(train_data_dict["input"], batch_index, t_past)
            label=train_data_dict["label"][batch_index]

            optimizer.zero_grad()
            mu, logvar=model(input)
            loss=criterion(mu, label, torch.exp(logvar))
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        train_loss/=len(train_index_loader)
        writer.add_scalar("train_loss", train_loss, epoch)
        
        if epoch % save_interval == 0 or epoch == total_epochs-1:
            checkpoint_path=result_path/"checkpoints"/f"epoch_{epoch}"
            os.makedirs(checkpoint_path,exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path/"model.pt")

            model.eval()
            test_loss=0
            with torch.no_grad():
                for batch_index in test_index_loader:
                    input=create_sequence(test_data_dict["input"], batch_index, t_past)
                    label=test_data_dict["label"][batch_index]
                    mu, logvar=model(input)
                    loss=criterion(mu, label, torch.exp(logvar))
                    test_loss+=loss.item()
            test_loss/=len(test_index_loader)
            writer.add_scalar("eval_loss", test_loss, epoch)


            # モデル性能の可視化
            t_target_list=[50,100,150,200]
            for t_target in t_target_list:
                with torch.no_grad():
                    t_start=t_target-t_future+1
                    t_list=torch.arange(t_start,t_target+1).to(device)
                    # print("test_data_dict['input'].shape",test_data_dict["input"].shape)
                    input=create_sequence(test_data_dict["input"], t_list, t_past)
                    label=test_data_dict["label"][t_list]
                    mu, logvar=model(input)
                    
                    mu=mu.cpu().numpy()
                    std=np.sqrt(np.exp(logvar.cpu().numpy()))

                    t_past_list=np.arange(t_target-t_past+1,t_target+1)
                    t_future_list=np.arange(t_target+1,t_target+t_future+1)

                # 予測の可視化
                axes=list[plt.Axes]
                labels=["open","high","low","close","volume"]
                fig,axes=plt.subplots(len(labels),1,figsize=(8,8))
                for i,ax in enumerate(axes):
                    # ax.plot(t_past_list,input[-1,:,i],color="black")
                    ax.plot(
                        np.concatenate([t_past_list,t_future_list]),
                        np.concatenate([input[-1,:,i],label[:,i]]),
                        label="actual",color="black"
                    )
                    ax.plot(t_future_list,mu[:,i],label="pred",color="red")
                    ax.fill_between(t_future_list,mu[:,i]-std[:,i],mu[:,i]+std[:,i],alpha=0.3,color="red")
                    ax.legend(loc="upper left")
                    ax.set_title(labels[i])
                imgpath=checkpoint_path/"images"
                os.makedirs(imgpath,exist_ok=True)
                fig.tight_layout()
                fig.savefig(imgpath/f"result_eval_t{t_target}.png")
                plt.close()

    writer.close()

if __name__=="__main__":
    main()
