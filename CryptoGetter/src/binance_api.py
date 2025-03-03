from datetime import datetime
import pandas as pd
import requests

from .crypto_api_abc import CryptoApiABC


class BinanceApi(CryptoApiABC):

    def __init__(self):
        super().__init__(
            url="https://api.binance.com/api/v3/klines",
            headers={
                "Accept": "application/json"
            }
        )

        self.request_columns=[
            "open time","O","H","L","C","volume",
            "close time","quote","trades count",
            "base volume","quote volume","unused"
        ]


        self.source_columns=["open time","O","H","L","C","volume","close time"]
        self.target_columns=["open-time","open","high","low","close","volume","close-time"]

    def get_ohlcv(self, symbol: str, interval: str, start_time: str, end_time: str) -> pd.DataFrame:

        result_db=pd.DataFrame()
        
        start_time_int=self.__strdate2ms(start_time)
        end_time_int=self.__strdate2ms(end_time)

        request_count=0
        while start_time_int <= end_time_int:

            limit=self.__request_data_limit(start_time_int, end_time_int, interval, max_limit=1000)
            if limit <= 0: break

            query={
                "symbol":symbol,
                "interval":interval,
                "startTime":start_time_int,
                "limit":limit
            }
            print(f"request_count: {request_count}, query: {query}")
            response=requests.get(url=self.url, headers=self.headers, params=query)
            request_count+=1

            response_db=pd.DataFrame(
                response.json(),
                columns=self.request_columns
            )

            result_db=pd.concat([result_db, response_db])

            start_time_int = int(response_db.iloc[-1]["close time"]) + 1 #1ms後を次のリクエストにわたす

        result_db=self.__change_column_names(result_db, self.target_columns, self.source_columns)
        result_db["open-time"]=result_db["open-time"].map(self.__ms2strdate)
        result_db["close-time"]=result_db["close-time"].map(self.__ms2strdate)


        return result_db[self.target_columns]


    def __interval2ms(self, interval: str) -> int:
        """
        時間軸をミリ秒に変換する
        """
        
        # 1dの時
        if interval == "1d":
            return 24 * 60 * 60 * 1000
        # 1hの時
        elif interval == "1h":
            return 60 * 60 * 1000
        # 1mの時
        elif interval == "1m":
            return 60 * 1000
        

    def __strdate2ms(self, str_date: str) -> int:
        """
        文字列の日付をミリ秒に変換する
        """
        dt = datetime.strptime(str_date, '%Y/%m/%d')
        return int(dt.timestamp()*1000)
    

    def __change_column_names(self, db: pd.DataFrame, target_columns: list, source_columns: list) -> pd.DataFrame:
        """
        カラム名を変換する
        :param db: データフレーム
        :param target_columns: 変換後のカラム名
        :param source_columns: 変換前のカラム名
        :return: 変換後のデータフレーム
        """
        return db.rename(columns=dict(zip(source_columns, target_columns)))


    def __ms2strdate(self, ms: int) -> str:
        """
        ミリ秒を文字列の日付に変換する
        """
        return datetime.fromtimestamp(ms/1000)

    def __request_data_limit(self, start_time: datetime, end_time: datetime, interval: str,max_limit:int) -> int:
        """
        リクエストデータサイズを計算する
        """
        data_size=(end_time-start_time)/self.__interval2ms(interval) + 1 #取ってくるデータサイズ
        limit=data_size if data_size < max_limit else max_limit
        return int(limit)
