from pathlib import Path
PARENT=Path(__file__).parent
ROOT=Path(__file__).parent.parent.parent
from datetime import timedelta, datetime
import yaml
from stable_baselines3 import PPO
import torch
import numpy as np
import pandas as pd

from ..web_api.coindesk_api import CoinDeskAPI


def get_action_prob(policy:PPO, obs:np.ndarray):
    """
    policyのaction_probを返す
    """
    with torch.no_grad():
        obs_tr=torch.Tensor(obs).unsqueeze(0).to(policy.policy.device)
        distribute=policy.policy.get_distribution(obs_tr)
        probs_tr:torch.Tensor=distribute.distribution.probs
        probs=probs_tr.detach().cpu().numpy()
    return probs



class CryptoAI:
    def __init__(self):
        self.columns=["open", "high", "low", "close", "volume"]

    def predict(self, model_name:str, to_timestamp:int):

        conf=yaml.safe_load(open(ROOT/"ai_model"/model_name/"conf.yml", "r", encoding="utf-8"))
        model=PPO.load(ROOT/"ai_model"/model_name/"result"/conf["model_name"])

        original_sequence, observation=self.__get_observation(
            input_sequence_length=conf["sequence_length"], to_timestamp=to_timestamp,
            fsym=conf["fsym"], tsym=conf["tsym"], interval=conf["interval"], trade_term=conf["trade_term"]
        )

        action_prob=get_action_prob(model, observation)[0]

        response={
            "long_prob": action_prob[0],
            "short_prob": action_prob[1],
            "input_sequence": self.__drop_latest_hlcv(
                original_sequence
            ).to_dict(orient="records"),
        }

        return response

        

    def __get_observation(
            self, input_sequence_length:int, to_timestamp:int,
            fsym:str, tsym:str, interval:str, trade_term:int
        ):

        # 先月末のタイムスタンプ. これ以前のデータで正規化する
        to_timestamp_norm = self.__get_last_timestamp_of_previous_month(to_timestamp)
        norm_max = self.__get_norm_params(fsym, tsym, interval, to_timestamp_norm, trade_term)


        # 入力データの取得
        original_sequence=CoinDeskAPI.get_ohlcv(fsym, tsym, interval, to_timestamp, input_sequence_length-1)
        observation=(original_sequence[self.columns]/norm_max).values.flatten()[:-4]

        def log():
            print(f"\033[33m--- START log: CryptoAI.__get_observation "+"-"*50+"\033[0m") 
            print("original_sequence:")
            print(original_sequence)
            print("\nnorm_max:")
            print(norm_max)
            print("\nobservation:")
            print(observation)
            print(f"\033[33m--- END log: CryptoAI.__get_observation --"+"-"*50+"\033[0m\n") 
        # log()
        return original_sequence, observation


    def __drop_latest_hlcv(self, sequence:pd.DataFrame):
        """
        最新のhlcvを削除する
        """
        columns = ["high", "low", "close", "volume"]
        for col in columns:
            sequence.at[sequence.index[-1], col] = None  # atを使用して元のDataFrameを変更
        return sequence
    

    def __get_norm_params(self, fsym:str, tsym:str, interval:str, to_timestamp:int, sequence_length:int):
        """
        過去のシーケンスデータから正規化パラメータを取得する
        :return norm_max <pd.Series>
        """

        past_sequence=CoinDeskAPI.get_ohlcv(fsym, tsym, interval, to_timestamp, sequence_length)
        norm_max=past_sequence[self.columns].max()

        return norm_max


    def __get_last_timestamp_of_previous_month(self, timestamp: int) -> int:
        """
        指定されたUnixタイムスタンプから先月末のタイムスタンプを取得する関数。

        :param timestamp: Unixタイムスタンプ（例: 1696118400）
        :return: 先月末のUnixタイムスタンプ
        """
        # Unixタイムスタンプをdatetimeオブジェクトに変換
        dt = datetime.fromtimestamp(timestamp)

        # 現在の月の1日を取得
        first_day_of_current_month = dt.replace(day=1)

        # 先月の最終日を取得
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)

        # 先月末のdatetimeオブジェクトをUnixタイムスタンプに変換して返す
        return int(last_day_of_previous_month.timestamp())