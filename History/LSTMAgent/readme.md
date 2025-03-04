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
