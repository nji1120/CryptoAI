from pathlib import Path
PARENT_DIR=Path(__file__).parent
MODEL_ROOT=Path(__file__).parent.parent.parent
import sys
sys.path.append(str(MODEL_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.data_formatter import DataFormatter



def main():

    datapath=MODEL_ROOT/"data/test.csv"
    data=pd.read_csv(datapath)[["open", "high", "low", "close", "volume"]].values

    t_dict={"past":30,"future":10,"base":26}
    # t_dict={"past":30,"future":30,"base":0}

    t_past=t_dict["past"]
    t_future=t_dict["future"]
    t_base=t_dict["base"]

    data_formatter=DataFormatter()
    formatted_data=data_formatter.get_formatted_data(t_past=t_past, t_future=t_future, data=data)

    line_width=3
    axes:list[plt.Axes]
    fig,axes=plt.subplots(5,1, figsize=(10,10))
    titles=["open", "high", "low", "close", "volume"]
    for i,ax in enumerate(axes):
        ax.plot(formatted_data["input"][:,i], label="current")
        ax.plot(formatted_data["label"][:,i], label="future")

        y_min,_=ax.get_ylim()
        value_t=formatted_data["input"][t_base,i]
        ax.annotate(
            '',  # テキストなし
            xy=(t_base, value_t),      # 矢印の先端
            xytext=(t_base, y_min),         # 矢印の始点
            arrowprops=dict(arrowstyle="->", color="blue", linewidth=line_width)
        )

        value_t_future=formatted_data["input"][t_base+t_future,i]
        ax.annotate(
            '',  # テキストなし
            xy=(t_base+t_future, value_t_future),      # 矢印の先端
            xytext=(t_base+t_future, y_min),         # 矢印の始点
            arrowprops=dict(arrowstyle="->", color="orange", linewidth=line_width)
        )    

        
        ax.annotate(
            '',  # テキストなし
            xy=(t_base, value_t_future),      # 矢印の先端
            xytext=(t_base, value_t),         # 矢印の始点
            arrowprops=dict(arrowstyle="->", color="red", linewidth=line_width)
        )

        ax.set_title(titles[i])
        ax.legend()

    fig.tight_layout()

    plt.savefig(PARENT_DIR/"test_data_formatter.png")


if __name__=="__main__":
    main()
