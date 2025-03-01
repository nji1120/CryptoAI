# MovingAvgAgent
入力に移動平均を使ったバージョン

## how to use
### モデルの設定
rootの`conf.yml`を変更する  
(いちいち専用のディレクトリにconfを作る必要はない)  

### モデルの学習
```bash
python train_cryptoai.py --train
```
実行によって, `output/{datetime.now}` にモデルが保存される  
conf.ymlもコピーされる  

### モデルのテスト
```bash
python train_cryptoai.py --conf {conf.ymlのパス}
```
`--train`オプションをつけないとテストになる  
trainで保存されたconfのパスを渡してテストする  

## SimpleAgentからの差分
### agent
- 報酬を累積確定利益率から確定利益率に変更  
  これによって、即時報酬としての意味合いが強くなった  
  つまり、closeしたときに瞬間的に大きな報酬が入るようになった  
  ↓  
  結果として、value_lossが発散しなくなった  
  累積だと発散するのは、それはそう  
  なぜなら、累積報酬は現在の入力状態から知ることはできないから  
   

### env
- 入力にmoving averageを追加  
  これによって、過去の情報の動きも踏まえて判断できるようになった  
