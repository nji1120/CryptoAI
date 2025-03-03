from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

class CryptoApiABC(ABC):
    def __init__(self, url: str, headers: dict):
        self.url = url
        self.headers = headers

    @abstractmethod
    def get_ohlcv(self, symbol: str, interval: str, start_time: str, end_time: str) -> pd.DataFrame:
        """
        OHLCVを取得する
        :param symbol: 通貨ペア. BTCUSDT, ETHUSDT, etc...
        :param interval: 時間軸. 1d, 1h, 1m, etc...
        :param start_time: 開始時間. 2024/01/01
        :param end_time: 終了時間. 2024/01/01
        :return: OHLCV [open-time, open, high, low, close, volume, close-time]
        """
        pass


