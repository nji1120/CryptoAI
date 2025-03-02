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
            self, T:int, sequence_length:int,
            agent: CryptoAgent, datapath:str
        ):
        """
        :param T: 1epochの取引日数
        :param sequence_length: シーケンスの長さ
        :param agent: agent
        :param datapath: データパス
        """

        super(CryptoEnv, self).__init__()

        self.action_space=spaces.Discrete(2) # 2つのアクション{LONG, SHORT}

        """
        ohlcv0, ohlcv1, ohlcv2, ... , ohlcv(N-1), open
        """
        observation_space_size=5 * (sequence_length-1) + 1 # 今はopenの情報だけ使う
        self.observation_space=spaces.Box(low=-1000, high=1000, shape=(observation_space_size,), dtype=np.float32) 

        self.agent=agent #agent
        self.T=T
        self.sequence_length=sequence_length
        self.original_df=self.__load_crypto_data(datapath)

        self.start_idx:int
        self.current_idx:int
        self.past_max:pd.Series


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
        self.past_max=self.__get_past_max()

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

        price_open=current_data_norm["O"]
        price_close=current_data_norm["C"]
        realized_pnl_rate=self.agent.act(action_idx, price_open, price_close)
        reward=realized_pnl_rate # 報酬は即時利益率そのもの

        observation=self.__get_observation()

        done = ( self.current_idx >= self.start_idx+self.T ) # Tstep経過で終了
        info={}

        self.current_idx+=1

        return observation, reward, done, False ,info
    

    def __get_past_sequence(self):
        """
        過去のシーケンスを取得
        """
        past_df=self.original_df.iloc[self.current_idx-self.sequence_length+1:self.current_idx]
        return past_df
    

    def __get_observation(self):
        """
        観察の取得.
        return: o,h,l,c,v, mv1_o,mv1_h,mv1_l,mv1_c,mv1_v,, ... , open_n
        """

        current_data=self.get_current_data()
        past_sequence=self.__get_past_sequence()
        sequence=pd.concat([past_sequence, current_data.to_frame().T]) #現在のohlcvまでのシーケンス
        sequence_norm=self.__normalize(sequence)

        observation=sequence_norm.values.flatten()[:-4] #現在のopen以外は捨てる

        def log_obs():
            print("-"*50)
            print("step:",self.current_idx)
            print("current_data\n",current_data)
            print("sequence\n",sequence)
            print("sequence_norm\n",sequence_norm)
            print("observation\n",observation, "\nshape:",observation.shape)
        # log_obs()

        return observation.astype(np.float32)
        

    def __load_crypto_data(self, datapath:str):
        """
        暗号通貨のデータを読み込む
        """
        columns=["O", "H", "L", "C", "volume"]
        df=pd.read_csv(datapath)[columns]
        return df


    def __get_start_idx(self):
        """
        取引開始indexのランダム取得
        """
        idx_max=len(self.original_df)-self.T
        idx_min=self.T+1 # T分の過去は正規化データ取得に使うため
        return np.random.randint(idx_min, idx_max)
    
    def __get_past_max(self):
        """
        過去Tstepの最大値を計算
        """
        past_df=self.original_df.iloc[self.current_idx-self.T:self.current_idx]
        past_max=past_df.max()
        return past_max

    def __normalize(self, x:pd.Series) -> pd.Series:
        """
        正規化
        """
        return x/self.past_max
    
    def get_current_data(self):
        """
        現在のデータを取得
        """
        return self.original_df.iloc[self.current_idx]
