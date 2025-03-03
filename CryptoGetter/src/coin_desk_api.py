from datetime import datetime
import pandas as pd
import requests

from .crypto_api_abc import CryptoApiABC


class CoinDeskApi(CryptoApiABC):

    def __init__(self):
        super().__init__(
            url="https://min-api.cryptocompare.com/data/v2",
            headers={
                "Accept": "application/json"
            }
        )

        # 一般的に, timeだけの場合はopen-timeを表す
        # volumetoは通過の取引量, volumefromはcryptoの取引量
        self.request_columns=[
            "time","open","high","low","close","volumefrom","volumeto",
            "conversionType","conversionSymbol"
        ]

        self.source_columns=[
            "time","open","high","low","close","volumeto"
        ]
        self.target_columns=[
            "open-time","open","high","low","close","volume"
        ]

    def get_ohlcv(self, symbol: str, interval: str, start_time: str, end_time: str) -> pd.DataFrame:

        result_db=pd.DataFrame()
        start_time_int=(self.__strdate2timestamp(start_time)) #日本時間をGMT時間に変換
        end_time_int=(self.__strdate2timestamp(end_time)) #日本時間をGMT時間に変換

        request_count=0
        while start_time_int <= end_time_int:

            limit=self.__request_data_limit(start_time_int, end_time_int, interval, max_limit=2000)
            if limit <= 0: break

            fsym, tsym=self.__split_symbol(symbol)
            to_ts=start_time_int+limit*self.__interval2seconds(interval) #to_tsまでのlimitコのデータをとってくる
            interval_query=self.__interval2query(interval)
            query={
                "fsym":fsym,
                "tsym":tsym,
                "limit":limit,
                "toTs":to_ts
            }
            print(f"request_count: {request_count}, query: {query}")
            response=requests.get(url=self.url+f"/{interval_query}", headers=self.headers, params=query)
            request_count+=1

            response_db=pd.DataFrame(
                response.json()["Data"]["Data"],
                columns=self.request_columns
            )

            result_db=pd.concat([result_db, response_db])

            start_time_int = int(response_db.iloc[-1]["time"]) + self.__interval2seconds(interval) #1interval後を次の開始時間とする

        result_db=self.__change_column_names(result_db, self.target_columns, self.source_columns)
        result_db["open-time"]=result_db["open-time"].map(datetime.fromtimestamp)

        return result_db[self.target_columns]

    def __split_symbol(self, symbol: str) -> tuple:

        if symbol.startswith("BTC"):
            return "BTC", symbol[3:]
        elif symbol.startswith("ETH"):
            return "ETH", symbol[4:]
        else:
            raise ValueError(f"Invalid symbol: {symbol}")
        
    def __request_data_limit(self, start_time: datetime, end_time: datetime, interval: str,max_limit:int) -> int:
        """
        リクエストデータサイズを計算する
        """
        data_size=(end_time-start_time)/self.__interval2seconds(interval) + 1 #取ってくるデータサイズ
        limit=data_size if data_size < max_limit else max_limit
        return int(limit)


    def __strdate2timestamp(self, str_date: str) -> int:
        return int(datetime.strptime(str_date, "%Y/%m/%d").timestamp())
    
    def __interval2seconds(self, interval: str) -> int:
        if interval == "1d":
            return 24 * 60 * 60
        elif interval == "1h":
            return 60 * 60
        elif interval == "1m":
            return 60


    def __jpy2gmt_timestamp(self, jpy_timestamp: int) -> int:
        """
        日本時間をGMT時間に変換する
        """
        return jpy_timestamp - 9 * 60 * 60
    
    def __gmt2jpy_timestamp(self, gmt_timestamp: int) -> int:
        """
        GMT時間を日本時間に変換する
        """
        return gmt_timestamp + 9 * 60 * 60


    def __interval2query(self, interval: str) -> str:
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


    def __change_column_names(self, db: pd.DataFrame, target_columns: list, source_columns: list) -> pd.DataFrame:
        """
        カラム名を変換する
        :param db: データフレーム
        :param target_columns: 変換後のカラム名
        :param source_columns: 変換前のカラム名
        :return: 変換後のデータフレーム
        """
        return db.rename(columns=dict(zip(source_columns, target_columns)))
