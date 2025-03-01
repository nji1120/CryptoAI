from gymnasium import Env, spaces
import numpy as np
import pandas as pd

from .crypto_agent import CryptoAgent

class CryptoEnv(Env):
    """
    crypto取引の環境
    """
    def __init__(self, T:int, alpha_realized:float, alpha_unrealized:float, agent: CryptoAgent, datapath:str):
        """
        :param T: 1epochの取引日数
        :param alpha_realized: 報酬計算に使う確定利益率の係数
        :param alpha_unrealized: 報酬計算に使う未確定利益率の係数
        :param agent: agent
        :param datapath: データパス
        """

        super(CryptoEnv, self).__init__()

        self.action_space=spaces.Discrete(4) # 4つのアクション{LONG, SHORT, CLOSE, STAY}

        # 8つの観察{open, high, low, close, volume, lot_hold, price_hold, position_hold}
        self.observation_space=spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32) 

        self.agent=agent #agent
        self.T=T
        self.alpha_realized=alpha_realized
        self.alpha_unrealized=alpha_unrealized
        self.original_df=self.__load_crypto_data(datapath)

        self.start_idx:int
        self.current_idx:int
        self.past_mean:pd.Series


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
        self.past_mean=self.__get_past_mean()

        # 初期observationの取得
        current_data=self.original_df.iloc[self.current_idx]
        current_data_norm=self.__normalize(current_data)
        observation=self.__get_observation(current_data_norm)

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
        current_data=self.__get_current_data()
        current_data_norm=self.__normalize(current_data)

        price_close=current_data_norm["C"]
        self.agent.act(action_idx, price_close)
        reward=self.__calculate_reward(price_close)

        observation=self.__get_observation(current_data_norm)

        done = ( self.current_idx >= self.start_idx+self.T )
        info={}

        self.current_idx+=1

        return observation, reward, done, False ,info
    
    
    def __get_observation(self, current_data_norm:pd.Series):
        """
        観察の取得.
        return: o,h,l,c,v, lot_hold, price_hold, position_hold
        """
        observation=np.concatenate([current_data_norm.values, self.agent.holdings])
        return observation.astype(np.float32)


    def __calculate_reward(self, price_close:float):
        """
        報酬の計算
        """
        # 未確定利益率
        unrealized_pnl_rate=self.agent.calculate_unrealized_pnl_rate(price_close)

        # 累積確定利益率 + 未確定利益率
        reward=self.alpha_realized*self.agent.realized_pnl_rate + self.alpha_unrealized*unrealized_pnl_rate
        return reward


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
        idx_min=self.T+1 # T分の過去は平均算出に使うため
        return np.random.randint(idx_min, idx_max)
    
    def __get_past_mean(self):
        """
        過去Tstepの平均を計算
        """
        past_df=self.original_df.iloc[self.current_idx-self.T:self.current_idx]
        past_mean=past_df.mean()
        return past_mean

    # def __get_past_minmax(self):
    #     """
    #     ※ これだと範囲外のデータが負になる可能性があるため不適
    #     過去Tstepの最小値と最大値を計算
    #     これで正規化する
    #     """
    #     past_df=self.original_df.iloc[self.current_idx-self.T:self.current_idx]
    #     past_min=past_df.min()
    #     past_max=past_df.max()
    #     return past_min, past_max
    
    def __normalize(self, x:pd.Series) -> pd.Series:
        """
        正規化
        """
        return x/self.past_mean
    
    def __get_current_data(self):
        """
        現在のデータを取得
        """
        return self.original_df.iloc[self.current_idx]
