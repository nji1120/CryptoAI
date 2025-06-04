from pathlib import Path
PARENT_DIR=Path(__file__).parent
MODEL_ROOT=Path(__file__).parent.parent.parent
import sys
sys.path.append(str(MODEL_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.data_formatter import DataFormatter



def main():

    datapath=MODEL_ROOT.parent/"data/test.csv"
    data=pd.read_csv(datapath)[["open", "high", "low", "close", "volume"]].values

    t_dict={"past":30,"future":10,"base":30}

    s_past=t_dict["past"]
    s_future=t_dict["future"]
    t_base=t_dict["base"]

    data_formatter=DataFormatter()
    formatted_data=data_formatter.get_formatted_data(
        data=data, 
        ts=[t_base for _ in range(s_future)], 
        s_past=s_past, 
        s_futures=[(i+1) for i in range(s_future)],
        is_normalize=True
    )
    print(f"formatted_data['input'].shape: {formatted_data['input'].shape}")
    print(f"formatted_data['target'].shape: {formatted_data['target'].shape}")

    line_width=3
    axes:list[plt.Axes]
    fig,axes=plt.subplots(5,1, figsize=(8,8))
    titles=["open", "high", "low", "close", "volume"]
    for i,ax in enumerate(axes):
        past_indices=np.arange(t_base-s_past+1,t_base+1,1)
        future_indices=np.arange(t_base+1,t_base+s_future+1,1)
        ax.plot(past_indices,formatted_data["input"][0,:,i], label="current")
        ax.plot(future_indices,formatted_data["target"][:,i], label="future")
        ax.set_title(titles[i])
        ax.legend(loc="upper left")

    fig.tight_layout()

    plt.savefig(PARENT_DIR/"test_data_formatter.png")


if __name__=="__main__":
    main()
