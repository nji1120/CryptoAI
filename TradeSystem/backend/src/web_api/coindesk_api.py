import requests
import pandas as pd
import datetime


BASE_URL="https://min-api.cryptocompare.com/data/v2"
HEADERS={
    "Accept": "application/json"
}
REQUEST_COLUMNS=[
    "time","open","high","low","close","volumefrom","volumeto",
    "conversionType","conversionSymbol"
]
SOURCE_COLUMNS=[
    "time","open","high","low","close","volumeto"
]
TARGET_COLUMNS=[
    "open-time","open","high","low","close","volume"
]

class CoinDeskAPI:
    @staticmethod
    def get_ohlcv(fsym:str, tsym:str, interval:str, to_timestamp:int, limit:int):
        """
        :param limit <= 2000
        """

        interval_query=CoinDeskAPI.interval2query(interval)
        query={
            "fsym":fsym,
            "tsym":tsym,
            "limit":limit,
            "toTs":to_timestamp
        }

        response=requests.get(BASE_URL+"/"+interval_query, params=query, headers=HEADERS)
        response_pd=pd.DataFrame(response.json()["Data"]["Data"], columns=REQUEST_COLUMNS)

        response_pd=CoinDeskAPI.change_column_names(
            response_pd,
            TARGET_COLUMNS,
            SOURCE_COLUMNS
        )
        response_pd["open-time"]=response_pd["open-time"].map(datetime.datetime.fromtimestamp)

        return response_pd[TARGET_COLUMNS]


    @staticmethod
    def interval2seconds(interval: str) -> int:
        """
        時間間隔を秒に変換する
        """
        if interval == "1d":
            return 86400
        elif interval == "1h":
            return 3600
        elif interval == "1m":
            return 60
        else:
            raise ValueError(f"Invalid interval: {interval}")


    @staticmethod
    def interval2query(interval: str) -> str:
        """
        時間間隔をクエリに変換する
        """
        if interval == "1d":
            return "histoday"
        elif interval == "1h":
            return "histohour"
        elif interval == "1m":
            return "histominute"
        else:
            raise ValueError(f"Invalid interval: {interval}")


    @staticmethod
    def change_column_names(db: pd.DataFrame, target_columns: list, source_columns: list) -> pd.DataFrame:
        """
        カラム名を変換する
        :param db: データフレーム
        :param target_columns: 変換後のカラム名
        :param source_columns: 変換前のカラム名
        :return: 変換後のデータフレーム
        """
        return db.rename(columns=dict(zip(source_columns, target_columns)))
