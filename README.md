# kaggle-base
Template directory for datascience competitions.
Data is saved in PostgreSQL on Dockcer container and the data is reproducibule/reusable.

# Usage

## Step0. Pull/Build Docker image
```
make pull
```
or 
```
make build
```

## Step1. Start up jupyter notebook
```
make jupyter
```

## Run the script
```
make run python src/foo.py
```

## flake8
```
make check
```

# References
### [1.データ分析コンペにおいて 特徴量管理に疲弊している全人類に伝えたい想い][1] 
まさに特徴量管理に疲弊していたときに見つけたスライド。すごくわかりやすいです。
### [2.Kaggleで使えるFeather形式を利用した特徴量管理法][2]
クラスの書き方が参考になります。
### [3.flowlight0's directory][3]


[1]:https://speakerdeck.com/takapy/detafen-xi-konpenioite-te-zheng-liang-guan-li-nipi-bi-siteiruquan-ren-lei-nichuan-etaixiang-i
[2]:https://amalog.hateblo.jp/entry/kaggle-feature-management
[3]:https://github.com/flowlight0/talkingdata-adtracking-fraud-detection
