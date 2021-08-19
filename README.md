# NetworkGenerate

## 生成モデル
### train

- 最初の学習時+データセットの作り直し、前処理を行う際は``--preprocess``を指定
- 学習済みの分類器を用いてfeedbackを行う場合は、``--classifier``を指定

```
python train.py [--preprocess --classifier]
```

### eval

- 訓練後にネットワークを行い評価
- ``eval_result``に結果をはく

```
python eval.py
```

### tuning

- tuning時に実行.
- tune.pyのscript内, 関数内でparam範囲指定
- configのopt_epoch分繰り返す
- param/best_tune.ymlにbest_tuneを吐き出す
- データセットの作り直し、前処理を行う際は``--preprocess``を指定
- 学習済みの分類器を用いてfeedbackを行う場合は、``--classifier``を指定

```
python tune.py [--preprocess --classifier]
```

## 分類モデル
### train

- 最初の学習時+データセットの作り直し、前処理を行う際は``--preprocess``を指定
- 結果は``classifier_train_result``を確認

```
python classifier_train.py [--preprocess]
```

### tuning

- tuning時に実行.
- classifier_tune.pyのscript内, 関数内でparam範囲指定
- configのopt_epoch分繰り返す
- param/classifier_best_tune.ymlにbest_tuneを吐き出す
- データセットの作り直し、前処理を行う際は``--preprocess``を指定

```
python classifier_tune.py [--preprocess]
```

### 全体の実行

- tune, train, eval, visualizeまでまとめて実行

'''
nohup python -u main.py --condition --train --eval --visualize --type generate_0 generate_1 generate_2 twitter_pickup --preprocess --pair --tune > log &
'''