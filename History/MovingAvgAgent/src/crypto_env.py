from gymnasium import Env, spaces
import numpy as np
import pandas as pd

from .crypto_agent import CryptoAgent

class CryptoEnv(Env):
    """
    crypto取引の環境
    """
    def __init__(
            self, T:int, moving_avg_windows:list[int],
            alpha_realized:float, alpha_unrealized:float, beta_reward:float, 
            agent: CryptoAgent, datapath:str
        ):
        """
        :param T: 1epochの取引日数
        :param moving_avg_windows: 移動平均のwindow
        :param alpha_realized: 報酬計算に使う確定利益率の係数
        :param alpha_unrealized: 報酬計算に使う未確定利益率の係数
        :param beta_reward: 報酬計算に使う係数
        :param agent: agent
        :param datapath: データパス
        """

        super(CryptoEnv, self).__init__()

        self.action_space=spaces.Discrete(4) # 4つのアクション{LONG, SHORT, CLOSE, STAY}

        """
        ohlcv, mv1_ohlcv, mv2_ohlcv, ... , mvN_ohlcv, lot_hold, price_hold, position_hold
        """
        observation_space_size=5 + 5 * len(moving_avg_windows)+3
        self.observation_space=spaces.Box(low=-1000, high=1000, shape=(observation_space_size,), dtype=np.float32) 

        self.agent=agent #agent
        self.T=T
        self.moving_avg_windows=moving_avg_windows
        self.alpha_realized=alpha_realized
        self.alpha_unrealized=alpha_unrealized
        self.beta_reward=beta_reward
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
        realized_pnl_rate=self.agent.act(action_idx, price_close)
        reward=self.__calculate_reward(price_close, realized_pnl_rate)

        observation=self.__get_observation(current_data_norm)

        done = ( self.current_idx >= self.start_idx+self.T ) # Tstep経過で終了
        info={}

        self.current_idx+=1

        return observation, reward, done, False ,info
    
    
    def __get_observation(self, current_data_norm:pd.Series):
        """
        観察の取得.
        return: o,h,l,c,v, mv1_o,mv1_h,mv1_l,mv1_c,mv1_v, mv2_o,mv2_h,mv2_l,mv2_c,mv2_v, ... , mvN_o,mvN_h,mvN_l,mvN_c,mvN_v, lot_hold, price_hold, position_hold
        """
        mv_ohlcv_list=[]
        for window in self.moving_avg_windows:
            mv_ohlcv=self.__get_moving_average(window)
            mv_ohlcv_norm=self.__normalize(mv_ohlcv)
            mv_ohlcv_list+=list(mv_ohlcv_norm.values)

        observation=np.concatenate([
            current_data_norm.values, 
            mv_ohlcv_list, 
            self.agent.holdings
        ])

        return observation.astype(np.float32)


    def __get_moving_average(self, window:int):
        """
        移動平均の取得
        """
        data_window=self.original_df.iloc[self.current_idx-window:self.current_idx]
        moving_average=data_window.mean()
        return moving_average
        

    def __calculate_reward(self, price_close:float, realized_pnl_rate:float):
        """
        報酬の計算
        """
        # 未確定利益率
        unrealized_pnl_rate=self.agent.calculate_unrealized_pnl_rate(price_close)

        # 確定利益率 + 未確定利益率
        reward=self.alpha_realized*realized_pnl_rate + self.alpha_unrealized*unrealized_pnl_rate
        reward=self.beta_reward*reward
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
