import pandas as pd
import numpy as np

class DataFormatter():


    def get_formatted_data(self,t_past: int, t_future: int, data: np.ndarray):
        """
        :param t_past: 過去の時間ステップ数 (現在も含む)
        :param t_future: t_futureステップ先の未来を正解ラベルとする
        :param data: 
        """

        # 未来のデータ取得
        future_data=self.get_future_data(data, t_future)

        # 正規化パラメータ取得
        norm_params=self.get_norm_params(data, t_past)

        # 正規化
        norm_data=self.normalize(data, norm_params)
        future_norm_data=self.normalize(future_data, norm_params)

        # 範囲のsplit
        t_start=t_past-1
        t_end=data.shape[0]-t_future-1
        norm_data=norm_data[t_start:t_end, :]
        future_norm_data=future_norm_data[t_start:t_end, :]

        # データセット
        dataset={
            "input":norm_data,
            "label":future_norm_data
        }

        return dataset
        

    def normalize(self, data:np.ndarray, norm_params:np.ndarray) -> np.ndarray:
        """
        正規化を行う
        data/norm_params
        :param data: 正規化するデータ [sequence_length, feature_size]
        :param norm_params: 正規化パラメータリスト [sequence_length, feature_size]
        :return: norm_data: *data.shape
        """
        return data/norm_params
        

    
    def get_norm_params(self, data:np.ndarray, t_past:int) -> np.ndarray:
        """
        正規化パラメータリストを取得する.
        :param data: データリスト. 正規化するデータ
        :param t_past: 過去の時間ステップ数 (現在も含む)
        :return: norm_params: *data.shape
        """
        
        norm_param=np.zeros_like(data)-1 # 0割を防ぐために-1で初期化
        t_end=data.shape[0]
        for t in range(t_past-1, t_end, 1):
            t_p=t-t_past+1
            norm_param[t]=np.mean(data[t_p:t+1, :], axis=0)

        return norm_param
            
    
    def get_future_data(self, data:np.ndarray, t_future:int) -> np.ndarray:
        """
        正解ラベルとなる未来のデータを取得する.
        :param data: データリスト. 未来のデータ
        :param t_future: 未来の時間ステップ数
        :return: future_data: *data.shape
        """
    
        future_data=np.zeros_like(data)
        t_end=data.shape[0]-t_future-1
        for t in range(0, t_end, 1):
            # future_data[t]=data[t+t_future, :] #値データそのまま格納
            future_data[t]=data[t+t_future, :]-data[t, :] #差分データを格納

        return future_data