# NetworkGenerate

## train

- 最初の学習時+データセットの作り直し、前処理を行う際は``--preprocess``を指定

```
python train.py [--preprocess]
```

## tuning

- tuning時に実行.
- tune.pyのscript内, 関数内でparam範囲指定
- configのopt_epoch分繰り返す
- param/best_tune.ymlにbest_tuneを吐き出す
- データセットの作り直し、前処理を行う際は``--preprocess``を指定

```
python tune.py [--preprocess]
```
