
from enum import Enum
import numpy as np

class Action(Enum):
    LONG=0
    SHORT=1
    CLOSE=2
    STAY=3

class Position(Enum):
    LONG=1
    SHORT=-1

class CryptoAgent:
    def __init__(self, lot=1):
        self.lot=lot

        self.__holdings={} # {lot, price, position}
        self.cumlative_realized_pnl_rate=0 #累積確定利益率


    def reset(self):
        self.__holdings={}
        self.cumlative_realized_pnl_rate=0


    def act(self, action_idx, price_close):
        """
        :param action_idx: policy nnからの出力. 既にindexになっている
        :param price_close: 正規化済みの終値.
        :return: 確定利益率 (closeの場合のみ値が返る. それ以外は0)
        """
        action=self.actual_action(action_idx)
        realized_pnl_rate=0

        if action==Action.LONG or action==Action.SHORT: # 注文
            self.__holdings["lot"]=self.lot
            self.__holdings["price"]=price_close
            self.__holdings["position"]=self.__action2position(action)

        elif action==Action.CLOSE: # 決済
            realized_pnl_rate=self.__calculate_pnl_rate(price_close) # 確定利益率
            # print(self.__holdings, "close: ",price_close, "pnl: ",realized_pnl_rate)
            self.cumlative_realized_pnl_rate+=realized_pnl_rate # 確定利益率の累計
            self.__holdings={} # holdを空にする

        elif action==Action.STAY: # 無操作
            pass

        return realized_pnl_rate

    def actual_action(self, action_idx):
        """
        hold状況から実際のactionを返す
        """
        input_action=Action(action_idx)

        # 持っていない場合は注文
        if (input_action==Action.LONG or input_action==Action.SHORT) and not self.__is_hold():
            return input_action
        
        # 持っている場合は決済
        elif input_action==Action.CLOSE and self.__is_hold():
            return Action.CLOSE
        
        # それ以外は無操作
        else:
            return Action.STAY


    @property
    def holdings(self):
        """
        保有ポジション.
        nn入力用にいい感じにして返す
        """
        if len(self.__holdings)==0:
            return np.array([0, -1, 0])
        else:
            holdings=np.array([
                self.__holdings["lot"], 
                self.__holdings["price"], 
                self.__holdings["position"].value
            ])
            return holdings


    def calculate_unrealized_pnl_rate(self, price_close):
        """
        未確定利益率
        """
        return self.__calculate_pnl_rate(price_close)


    def __is_hold(self):
        """
        既にpositionを持っているかどうか
        :return True->持っている, False->持っていない
        """
        return not len(self.__holdings)==0

    def __action2position(self, action):
        """
        :param action: action
        :return: position
        """
        if action==Action.LONG:
            return Position.LONG
        elif action==Action.SHORT:
            return Position.SHORT
        else:
            raise ValueError("Invalid action")


    def __calculate_pnl_rate(self, price_close):
        """
        利益率計算. holdが無いときは0を返す
        :param price_close: 終値
        :return: pnl
        """
        if len(self.__holdings)==0:
            return 0

        position=self.__holdings["position"].value
        price_hold=self.__holdings["price"]
        pnl_rate=position*(price_close-price_hold)/price_hold
        return pnl_rate



