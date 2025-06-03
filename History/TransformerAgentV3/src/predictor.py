# モデルを使って未来を予測する予測器

import torch
import torch.nn as nn
import numpy as np


class Predictor():
    def __init__(self, model: nn.Module, device: str,s_past:int,s_future:int):
        self.model=model
        self.device=device
        self.s_past=s_past
        self.s_future=s_future


    def get_need_sequence(self, data:np.ndarray, t_current:int) -> np.ndarray:
        """
        予測に必要なデータを抽出する
        :param data: [total_timesteps x feature_dim]
        :param t_current: 予測したい時刻
        :return: [s_need x feature_dim]
        """
        t_start=t_current-self.s_need+1
        t_end=t_current+1 #listの特性上+1する

        return data[t_start:t_end]
    

    def create_ts_right(self, t_current:int) -> np.ndarray:
        """
        予測に必要な時系列の最近(=行列の右端)の時刻
        """
        ts_right_head=self.ts_right_head(t_current)
        ts_right_tail=t_current+1
        return np.arange(ts_right_head,ts_right_tail)


    def create_sequence(self, data: np.ndarray, ts_right: np.ndarray) -> torch.Tensor:
        """
        :param data: [total_timesteps x feature_dim]
        :param ts_right: 予測したい時刻 [batch]
        :return: sequence: [batch x t_past x feature_dim]
        """
        batch_size=len(ts_right)
        sequence_data=np.zeros((batch_size, self.s_past, data.shape[1]))
        for i,t in enumerate(ts_right):
            sequence_data[i]=data[t-self.s_past+1:t+1]
        return sequence_data
    

    def predict(self, x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """
        :param x: 最後のstepが最近のデータであれば良い. [any_sequence_length x feature_dim]
        :return mu: [s_future x feature_dim]
        """
        t_current=x.shape[0]-1
        ts_right=self.create_ts_right(t_current)
        sequence=self.create_sequence(x, ts_right)

        self.model.eval()
        mu:torch.Tensor
        logvar:torch.Tensor
        with torch.no_grad():
            sequence=torch.from_numpy(sequence).to(self.device).to(torch.float32)
            mu, logvar=self.model(sequence)
            mu=mu.cpu().numpy()
            std=np.sqrt(np.exp(logvar.cpu().numpy()))

        return mu,std
            

    @property
    def s_need(self) -> int:
        """
        予測に必要な時系列長さ
        """
        return self.s_past+self.s_future-1
    
    def ts_right_head(self, t_current:int) -> int:
        """
        直近の予測に必要な時系列の最近の時刻
        """
        return t_current-self.s_future+1
    
    