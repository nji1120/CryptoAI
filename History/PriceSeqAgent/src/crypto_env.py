from gymnasium import Env, spaces
import numpy as np
import pandas as pd

from .crypto_agent import CryptoAgent

class CryptoEnv(Env):
    """
    crypto取引の環境
    """
    def __init__(
            self, T:int, sequence_length:int,
            alpha_realized:float, alpha_unrealized:float, beta_reward:float, 
            agent: CryptoAgent, datapath:str
        ):
        """
        :param T: 1epochの取引日数
        :param sequence_length: シーケンスの長さ
        :param alpha_realized: 報酬計算に使う確定利益率の係数
        :param alpha_unrealized: 報酬計算に使う未確定利益率の係数
        :param beta_reward: 報酬計算に使う係数
        :param agent: agent
        :param datapath: データパス
        """

        super(CryptoEnv, self).__init__()

        self.action_space=spaces.Discrete(4) # 4つのアクション{LONG, SHORT, CLOSE, STAY}

        """
        ohlcv0, ohlcv1, ohlcv2, ... , ohlcvN, lot_hold, price_hold, position_hold
        """
        observation_space_size=5 * sequence_length + 3
        self.observation_space=spaces.Box(low=-1000, high=1000, shape=(observation_space_size,), dtype=np.float32) 

        self.agent=agent #agent
        self.T=T
        self.sequence_length=sequence_length
        self.alpha_realized=alpha_realized
        self.alpha_unrealized=alpha_unrealized
        self.beta_reward=beta_reward
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
        current_data=self.__get_current_data()
        current_data_norm=self.__normalize(current_data)

        price_close=current_data_norm["C"]
        realized_pnl_rate=self.agent.act(action_idx, price_close)
        reward=self.__calculate_reward(price_close, realized_pnl_rate)

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
        return: o,h,l,c,v, mv1_o,mv1_h,mv1_l,mv1_c,mv1_v, mv2_o,mv2_h,mv2_l,mv2_c,mv2_v, ... , mvN_o,mvN_h,mvN_l,mvN_c,mvN_v, lot_hold, price_hold, position_hold
        """

        current_data=self.__get_current_data()
        past_sequence=self.__get_past_sequence()
        sequence=pd.concat([past_sequence, current_data.to_frame().T])
        sequence_norm=self.__normalize(sequence)

        observation=np.concatenate([
            sequence_norm.values.flatten(), 
            self.agent.holdings
        ])

        def log_obs():
            print("-"*50)
            print("step:",self.current_idx)
            print("current_data\n",current_data)
            print("sequence\n",sequence)
            print("sequence_norm\n",sequence_norm)
            print("observation\n",observation)
        # log_obs()

        return observation.astype(np.float32)
        

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
    
    def __get_current_data(self):
        """
        現在のデータを取得
        """
        return self.original_df.iloc[self.current_idx]
