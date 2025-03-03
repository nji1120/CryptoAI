from pathlib import Path
ROOT=Path(__file__).parent.parent
import sys
sys.path.append(str(ROOT))
import os
import yaml
import datetime

from src.binance_api import BinanceApi
from src.coin_desk_api import CoinDeskApi

def main():
    with open(ROOT / "app" / "query.yml", "r", encoding="utf-8") as f:
        query=yaml.safe_load(f)

    if query["api"] == "binance":
        api=BinanceApi()
    elif query["api"] == "coindesk":
        api=CoinDeskApi()
    else:
        raise ValueError(f"Invalid API: {query['api']}")

    ohlcv=api.get_ohlcv(
        symbol=query["symbol"],
        interval=query["interval"],
        start_time=query["start_time"],
        end_time=query["end_time"]
    )

    save_path=ROOT/"output"/datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    with open(save_path/"query.yml", "w", encoding="utf-8") as f:
        yaml.dump(query, f, sort_keys=False, encoding="utf-8", allow_unicode=True)
    ohlcv.to_csv(save_path/"ohlcv.csv", index=False)


if __name__ == "__main__":
    main()

