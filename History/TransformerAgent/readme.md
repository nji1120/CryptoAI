# Transformer Agent
## 概要
TransformerEncoderで特徴を抽出するAgent  
Tansformerの出力をactorとcriticにぶち込む  
ちなみに, LSTMモデルでは上手くいかなかった  
各時刻のデータを不連続に注目できるAttentionならLSTMより性能がでるかもしれない  