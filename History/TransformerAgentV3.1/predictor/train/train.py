from pathlib import Path
PARENT_DIR=Path(__file__).parent
MODEL_ROOT=Path(__file__).parent.parent
PROJECT_ROOT=MODEL_ROOT.parent.parent.parent
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
from src.prob_transformer import ProbTransformer


def create_result_path(root:Path, model_name:str) -> Path:
    current_time=datetime.now().strftime("%Y%m%d_%H%M%S")
    return root/model_name/current_time

def creat_dataloader(total_timesteps:int,t_past:int,t_future:int,batch_size:int,shuffle:bool,generator:torch.Generator) -> torch.utils.data.DataLoader:
    index=np.arange(t_past,total_timesteps-t_future)
    return torch.utils.data.DataLoader(index, batch_size=batch_size, shuffle=shuffle,generator=generator)


def eval_model(model:ProbTransformer, data_formatter:DataFormatter, data:np.ndarray, s_past:int, s_future:int, device:torch.device) -> float:
    """
    ある特定の時系列でevalを行う
    """
    t_target=np.random.randint(s_past,len(data)-s_future)
    s_futures=np.arange(1,s_future+1)

    formatted_data=data_formatter.get_formatted_data(
        data=data,
        ts=[t_target for _ in range(s_future)],
        s_past=s_past,
        s_futures=s_futures,
        is_normalize=True
    )

    input_data=torch.from_numpy(formatted_data["input"]).to(device).to(torch.float32)
    ts=torch.from_numpy(s_futures).to(device).to(torch.int32)

    model.eval()
    mu, logvar=model.predict(input_data, ts)

    output={
        "input_indices":np.arange(t_target-s_past+1,t_target+1),
        "target_indices":np.arange(t_target+1,t_target+s_future+1),
        "input":input_data.cpu().numpy()[0],
        "target":formatted_data["target"],
        "mu":mu.cpu().numpy(),
        "std":np.sqrt(np.exp(logvar.cpu().numpy())),
    }

    return output
    

def visualize_result(output:dict, save_path:Path, filename:str) -> None:
    """
    結果を可視化する
    """
    input_indices=output["input_indices"]
    target_indices=output["target_indices"]
    sequence_past=output["input"]
    sequence_future=output["target"]
    sequence_predicted=output["mu"]
    sequence_std=output["std"]
    attrs=["open", "high", "low", "close", "volume"]

    axes:list[plt.Axes]
    fig,axes=plt.subplots(5,1,figsize=(8,8))
    for i,attr in enumerate(attrs):
        axes[i].plot(input_indices,sequence_past[:,i], label="past",color="black")
        axes[i].plot(target_indices,sequence_future[:,i], label="future",color="blue")
        axes[i].plot(target_indices,sequence_predicted[:,i], label="predicted",color="red")
        axes[i].fill_between(
            target_indices,
            sequence_predicted[:,i]-sequence_std[:,i],
            sequence_predicted[:,i]+sequence_std[:,i],
            alpha=0.3,
            color="red"
        )
        axes[i].legend()
    fig.tight_layout()
    save_path.mkdir(parents=True,exist_ok=True)
    fig.savefig(save_path/filename)
    plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=PARENT_DIR/"config.yml")
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
    transformer_conf=model_conf["transformer"]
    out_layer_conf=model_conf["out_layer"]
    model=ProbTransformer(
        sequence_length=transformer_conf["sequence_length"],
        input_feature_dim=transformer_conf["feature_dim"],
        future_time_dim=transformer_conf["future_time_dim"],
        transformer_hidden_dim=transformer_conf["hidden_dim"],
        transformer_feedforward_dim=transformer_conf["feedforward_dim"],
        transformer_n_heads=transformer_conf["n_heads"],
        transformer_n_layers=transformer_conf["n_layers"],
        transformer_dropout=transformer_conf["dropout"],
        output_n_layers=out_layer_conf["n_layers"],
        output_hidden_dim=out_layer_conf["hidden_dim"],
        output_dim=out_layer_conf["out_dim"],
    )
    model.to(device)
    print(model)


    # **************************************** データの準備 *************************************************
    # データ整形器の作成
    data_formatter=DataFormatter()

    # データの読み込み
    columns=["open", "high", "low", "close", "volume"]
    train_datapath=Path(PROJECT_ROOT/train_conf["datapath"]["train"])
    train_data=pd.read_csv(train_datapath)[columns].values

    test_datapath=Path(PROJECT_ROOT/train_conf["datapath"]["test"])
    test_data=pd.read_csv(test_datapath)[columns].values



    # **************************************** 学習の準備 *************************************************
    # 学習パラメータ
    total_epochs=train_conf["total_epochs"]
    batch_size=train_conf["batch_size"]
    learning_rate=train_conf["learning_rate"]
    save_interval=train_conf["save_interval"]
    loss_weight=train_conf["loss_weight"]
    eval_num=train_conf["eval_num"]

    # 損失関数
    criterion=torch.nn.GaussianNLLLoss()

    # optimizer
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

    # indexの作成
    s_past=model_conf["input_params"]["s_past"]
    s_future=model_conf["input_params"]["s_future"]

    train_index_loader=creat_dataloader(
        len(train_data),s_past,s_future,
        batch_size,True,torch.Generator(device=device)
    )
    test_index_loader=creat_dataloader(
        len(test_data),s_past,s_future,
        batch_size,False,torch.Generator(device=device)
    )

    # tensorboardの準備
    writer=SummaryWriter(log_dir=result_path/"logs")

    # 学習
    for epoch in tqdm(range(total_epochs)):
        model.train()
        train_loss=0
        batch_index:torch.Tensor
        for batch_index in train_index_loader:

            s_futures=np.random.randint(1,s_future+1,batch_index.shape[0])
            train_dataset=data_formatter.get_formatted_data(
                data=train_data,
                ts=batch_index.cpu().numpy(),
                s_past=s_past,
                s_futures=s_futures,
                is_normalize=True
            )

            input_data=torch.from_numpy(train_dataset["input"]).to(device).to(torch.float32)
            ts=torch.from_numpy(s_futures).to(device).to(torch.int32)
            target=torch.from_numpy(train_dataset["target"]).to(device).to(torch.float32)

            optimizer.zero_grad()
            mu, logvar=model(input_data, ts)
            loss:torch.Tensor=criterion(mu, target, torch.exp(logvar))
            loss*=loss_weight
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        train_loss/=len(train_index_loader)
        writer.add_scalar("train_loss", train_loss, epoch)
        
        if epoch % save_interval == 0 or epoch == total_epochs-1:
            checkpoint_path=result_path/"checkpoints"/f"epoch_{epoch}"
            os.makedirs(checkpoint_path,exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path/"model.pt")


            # ****************************** 全体eval **************************************
            model.eval()
            test_loss=0
            with torch.no_grad():
                for batch_index in test_index_loader:

                    s_futures=np.random.randint(1,s_future+1,batch_index.shape[0])
                    test_dataset=data_formatter.get_formatted_data(
                        data=test_data,
                        ts=batch_index.cpu().numpy(),
                        s_past=s_past,
                        s_futures=s_futures,
                        is_normalize=True
                    )

                    input_data=torch.from_numpy(test_dataset["input"]).to(device).to(torch.float32)
                    ts=torch.from_numpy(s_futures).to(device).to(torch.int32)
                    target=torch.from_numpy(test_dataset["target"]).to(device).to(torch.float32)

                    mu, logvar=model(input_data, ts)
                    loss:torch.Tensor=criterion(mu, target, torch.exp(logvar))
                    loss*=loss_weight
                    test_loss+=loss.item()

            test_loss/=len(test_index_loader)
            writer.add_scalar("eval_loss", test_loss, epoch)


            # ****************************** 性能の可視化 **************************************
            for i in range(eval_num):
                eval_dataset={"train":train_data,"test":test_data}
                for key,data in eval_dataset.items():
                    output=eval_model(model, data_formatter, data, s_past, s_future, device)
                    visualize_result(output, checkpoint_path/f"images/{key}", f"result_eval{i}.png")



if __name__=="__main__":
    main()