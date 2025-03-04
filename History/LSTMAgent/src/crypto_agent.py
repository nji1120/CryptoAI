
from enum import Enum

class Action(Enum):
    LONG=0
    SHORT=1

class Position(Enum):
    LONG=1
    SHORT=-1

class CryptoAgent:
    def __init__(self):
        self.cumlative_realized_pnl_rate=0 #累積確定利益率


    def reset(self):
        self.cumlative_realized_pnl_rate=0


    def act(self, action_idx, price_open, price_close):
        """
        :param action_idx: policy nnからの出力. 既にindexになっている
        :param price_open: 正規化済みの始値.
        :param price_close: 正規化済みの終値.
        :return: 確定利益率 (closeの場合のみ値が返る. それ以外は0)
        """
        action=Action(action_idx)
        realized_pnl_rate=self.__calculate_pnl_rate(
            self.__action2position(action).value,
            price_open,
            price_close
        )
        self.cumlative_realized_pnl_rate+=realized_pnl_rate

        return realized_pnl_rate


    def __calculate_pnl_rate(self, position,price_open, price_close):
        """
        :param position: position (LONG -> 1, SHORT -> -1)
        :param price_open: 正規化済みの始値.
        :param price_close: 正規化済みの終値.
        :return: 確定利益率 (closeの場合のみ値が返る. それ以外は0)
        """
        return position*(price_close-price_open)/price_open


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


