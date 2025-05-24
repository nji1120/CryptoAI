# LSTMAgent
networkをLSTMで時系列を捉えるようにしたモデル  

## StepTradeAgentからの差分

### env
- observationの変更  
  - before  
    - latestはopenのみ出力
  - after
    - latestはopen以外を0パディング

### model
- ネットワークの変更
  - before
    - もともとは64x2のActorとCriticだったっぽい  
    - 活性化関数はどちらもtanh
  - after
    - LSTMで抽出したh_lastをactorとcriticの入力にする  
      ActorとCriticは1層にしてる  
    - 活性化関数はactorがReLU, criticがtanh  
      actorはどうせsoftmax通すのでReLUが勾配消失しなくていい  
      criticはrewardが-1~1なのでtanhとしている  
      (rewardは即時利益率でありほぼ0なのでtanhでも勾配消失しないはず)  

### conf
モデルの設定が必要になるため, configをtrain, model, dataに分けた  


## モデル一覧
### LSTMPolicy
- 特徴
  LSTMのh_Tをactorとcriticの入力にするモデル  

- モデル性能  
  結果として, ただのMlpPolicyに時系列データをベクトルとしてぶち込むよりも性能が悪かった.  

- 考察  
  考えられる理由としては, cryptoデータは直近のデータよりも, 過去のとびとびのデータが大事になってくる可能性がある.  
  このため, 全ての時間データを均等に見れるMlpPolicyの方が性能が良かった可能性がある.

- 参考
  ||MlpPolicy|LSTMPolicy|
  |---|---|---|
  |reward_mean|0.6~0.7|0.5~0.55|

### SeqBidLSTMPolicy
- 特徴
  LSTMの全htをactorとcriticの入力にするモデル.  
  ここで, LSTMをbidirectonalにして, 両方向からデータを見れるようにしている.  
  LSTMPolicyの問題と思われる点を改善  

- 仮説  
  全ての時系列を均等にactorcriticに入力 & 時系列の特徴も埋め込んでいるので, MlpPolicyよりも性能が良くなるはず.  

- モデル性能  
  MlpPolicyに時系列データをベクトルとしてぶち込むよりも性能が良かった.  

- 考察  
  

