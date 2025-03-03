from pathlib import Path
ROOT=Path(__file__).parent.parent
import sys
sys.path.append(str(ROOT))

from datetime import datetime, timedelta

from src.crypto_ai.crypto_ai import CryptoAI


def test_crypto_ai():

    strdatetime="2025/3/3 9:00:00"
    
    dt=datetime.strptime(strdatetime,"%Y/%m/%d %H:%M:%S")
    crypto_ai=CryptoAI()
    response=crypto_ai.predict(
        model_name="20250302_153821",
        to_timestamp=dt.timestamp()
    )
    print(response)

if __name__ == "__main__":
    test_crypto_ai()

