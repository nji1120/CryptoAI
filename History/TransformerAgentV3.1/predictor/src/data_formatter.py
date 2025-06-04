import pandas as pd
import numpy as np

class DataFormatter():

    def get_formatted_data(self,data: np.ndarray, ts:list[int], s_past:int, s_futures:list[int], is_normalize:bool=True):
        """
        :param data: 全timestepのデータ. [timestep, feature_dim]
        :param ts: 入力する時刻のheadリスト
        :param s_past: 過去の時間ステップ数 (現在も含む)
        :param s_futures: t+s_futureステップ先の未来を正解ラベルとする
        """
        def debug_input():
            print(f"data: {data.shape}")
            print(f"ts: {ts}")
            print(f"s_past: {s_past}")
            print(f"s_futures: {s_futures}")
        # debug_input()

        # 現在のデータ取得
        current_sequence=self.get_current_sequence(data, ts, s_past)

        # 未来のデータ取得
        future_data=self.get_future_data(data, ts, s_futures)

        # 正規化パラメータ取得
        norm_params=self.get_norm_params(data, ts, s_past)

        # 正規化
        if is_normalize:
            current_sequence=self.normalize(current_sequence, np.expand_dims(norm_params, axis=1))
            future_data=self.normalize(future_data, norm_params)

        # データセット
        dataset={
            "input":current_sequence,
            "target":future_data
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
        return data/(norm_params+1e-8)
    


    def get_current_sequence(self, data:np.ndarray, ts:list[int], s_past:int) -> np.ndarray:
        """
        現在のデータを取得する.
        :param data: データリスト. 現在のデータ
        :param ts: 入力する時刻のheadリスト
        :param s_past: 過去の時間ステップ数 (現在も含む)
        """
        feature_dim=data.shape[1]
        batch_size=len(ts)
        current_sequence=np.zeros((batch_size, s_past, feature_dim))
        for i, t in enumerate(ts):
            current_sequence[i]=data[t-s_past+1:t+1, :]

        return current_sequence
        

    
    def get_norm_params(self, data:np.ndarray, ts:list[int], s_past:int) -> np.ndarray:
        """
        正規化パラメータリストを取得する.
        :param data: データリスト. 正規化するデータ
        :param ts: 入力する時刻のheadリスト
        :param s_past: 過去の時間ステップ数 (現在も含む)
        :return: norm_params: [batch_size, feature_dim]
        """
        
        feature_dim=data.shape[1]
        batch_size=len(ts)
        norm_prams=np.zeros((batch_size, feature_dim))

        for i, t in enumerate(ts):
            norm_prams[i]=np.mean(data[t-s_past+1:t+1, :], axis=0) #+1でt(現在)も含む

        return norm_prams
            
    
    def get_future_data(self, data:np.ndarray, ts:list[int], s_futures:list[int]) -> np.ndarray:
        """
        正解ラベルとなる未来のデータを取得する.
        :param data: データリスト. 未来のデータ
        :param ts: 入力する時刻のheadリスト
        :param s_futures: t+s_futureステップ先の未来を正解ラベルとする
        :return: future_data: [batch_size, feature_dim]
        """
    
        feature_dim=data.shape[1]
        batch_size=len(ts)
        future_data=np.zeros((batch_size, feature_dim))
        for i, t in enumerate(ts):
            future_data[i]=data[t+s_futures[i], :]

        return future_data