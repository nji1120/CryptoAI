from gymnasium import Env, spaces
import numpy as np
import pandas as pd

from .crypto_agent import CryptoAgent

class CryptoEnv(Env):
    """
    crypto取引の環境.
    毎stepトレードを行う環境 (=dayトレード)
    """
    def __init__(
            self, trade_term:int, sequence_length:int,
            agent: CryptoAgent, datapath:str, 
            is_test_log:bool=False
        ):
        """
        :param trade_term: 1epochの取引日数
        :param sequence_length: シーケンスの長さ
        :param agent: agent
        :param datapath: データパス
        :param is_test_log: テストログを出力するかどうか(コードのテスト用)
        """

        super(CryptoEnv, self).__init__()

        self.action_space=spaces.Discrete(2) # 2つのアクション{LONG, SHORT}
        
        # 今はopenの情報だけ使う. 未来のhigh, low, close, volumeは0で埋める
        # ohlcv0, ohlcv1, ohlcv2, ... , ohlcv(N-1), open, 0, 0, 0
        observation_space_size=5 * sequence_length
        self.observation_space=spaces.Box(low=-1000, high=1000, shape=(observation_space_size,), dtype=np.float32) 

        self.agent=agent #agent
        self.trade_term=trade_term
        self.sequence_length=sequence_length
        self.original_df=self.__load_crypto_data(datapath)

        self.start_idx:int
        self.current_idx:int
        self.past_price_max:float
        self.past_volume_max:float

        self.is_test_log=is_test_log

    def reset(self, seed:int=None):
        """
        環境のリセット
        :return observation: 初期観察
        :return info: 空の辞書
        """
        self.agent.reset()  # agentのリセット


        # 環境のリセット
        self.start_idx=self.__get_start_idx()
        self.current_idx=self.start_idx
        self.past_price_max, self.past_volume_max=self.__get_past_max()

        # 初期observationの取得
        observation=self.__get_observation()

        return observation, {}


    def step(self, action_idx:int):
        """
        :param action_idx: policy nnからの出力. 既にindexになっている
        :return observation: 観察
        :return reward: 報酬
        :return done: エピソード終了フラグ
        :return truncated: エピソード切り捨てフラグ(未使用)
        :return info: 空の辞書
        """
        current_data=self.get_current_data()
        current_data_norm=self.__normalize(current_data)

        price_open=current_data_norm["open"]
        price_close=current_data_norm["close"]
        reward=self.agent.act(action_idx, price_open, price_close) # 報酬は即時利益率そのもの

        observation=self.__get_observation()

        done = ( self.current_idx >= self.start_idx+self.trade_term ) # Tstep経過で終了
        info={}

        self.current_idx+=1

        return observation, reward, done, False ,info
    

    def __get_past_sequence(self):
        """
        過去のシーケンスを取得
        """
        past_df=self.original_df.iloc[self.current_idx-self.sequence_length:self.current_idx]
        return past_df
    

    def __get_observation(self):
        """
        前stepまでのobservationの取得.
        今のopenは入れない(どうせ前ステップのcloseと同じため)  
        LSTMでは0パディングしてたけど, それだとNNに入れたときに重みが意味なくなってしまうと思う. 
        Policy内でreshapeする
        return: ohlcv0, ohlcv1, ohlcv2, ... , ohlcv(N-1)
        """

        past_sequence=self.__get_past_sequence()
        sequence_norm=self.__normalize(past_sequence)

        observation=sequence_norm.values
        observation=observation.flatten()


        # -- コードのテスト用 ---
        def log_obs():
            print("-"*50)
            print("step:",self.current_idx)
            print("past_price_max:",self.past_price_max)
            print("past_sequence\n",past_sequence)
            print("sequence_norm\n",sequence_norm)
            print("observation\n",observation, "\nshape:",observation.shape)
        if self.is_test_log: log_obs()
        # -- コードのテスト用 ---


        return observation.astype(np.float32)
        

    def __load_crypto_data(self, datapath:str):
        """
        暗号通貨のデータを読み込む
        """
        columns=["open", "high", "low", "close", "volume"]
        df=pd.read_csv(datapath)[columns]
        return df


    def __get_start_idx(self):
        """
        取引開始indexのランダム取得
        """
        idx_max=len(self.original_df)-self.trade_term
        idx_min=self.trade_term+1 if self.trade_term>self.sequence_length else self.sequence_length+1
        return np.random.randint(idx_min, idx_max)
    
    def __get_past_max(self):
        """
        過去Tstepの最大値をとる
        """
        past_df=self.original_df.iloc[self.current_idx-self.trade_term:self.current_idx]
        past_price_max=np.max(past_df[["open", "high", "low", "close"]].values) #過去price(open, high, low, close全て)の最大値
        past_volume_max=np.max(past_df["volume"].values) #過去volumeの最大値
        return past_price_max, past_volume_max

    def __normalize(self, x:pd.DataFrame) -> pd.DataFrame:
        """
        正規化
        """
        x_norm=x.copy()
        x_norm[["open", "high", "low", "close"]]=x_norm[["open", "high", "low", "close"]]/self.past_price_max # priceは同じmaxで割る
        x_norm["volume"]=x_norm["volume"]/self.past_volume_max # volumeは過去のvolumeの最大値で割る

        return x_norm
    
    def get_current_data(self):
        """
        現在のデータを取得
        """
        return self.original_df.iloc[self.current_idx]
