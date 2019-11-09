# kaggle-base

- Template directory for datascience competitions.
- Data is saved in PostgreSQL on Dockcer container and the data is reproducibule/reusable.

## Usage

### Step0. Pull/Build Docker image

Recommended:

```sh
make pull
```

or

```sh
make build
```

### Step1. Start up jupyter notebook

```sh
make jupyter
```

- Copy token and acccess to localhost:${JUPYTER_PORT}

### Step2. Start up DB

```sh
make start-db
```

- Then you can access to localhost:${PGWEB_PORT} to view the database.
- You can access to localhost:${METABASE_PORT} to do the simple BI.

### Step3. Create Features

- Create all features.

```sh
make feature
```

- Specify a feature that will be created.

```sh
make feature FEATURE_NAME
```

## Commands

### isort, black

```sh
make format
```

### flake8, mypy

```sh
make check
```

### execute scripts

Recommended:

```sh
make shell
python xxx.py
```

or

```sh
make run python xxx.py
```


## References

### [1.データ分析コンペにおいて 特徴量管理に疲弊している全人類に伝えたい想い][1]

まさに特徴量管理に疲弊していたときに見つけたスライド。すごくわかりやすいです。

### [2.Kaggleで使えるFeather形式を利用した特徴量管理法][2]

クラスの書き方が参考になります。

### [3.flowlight0's directory][3]

### [4.upura's directory][4]

[1]:https://speakerdeck.com/takapy/detafen-xi-konpenioite-te-zheng-liang-guan-li-nipi-bi-siteiruquan-ren-lei-nichuan-etaixiang-i
[2]:https://amalog.hateblo.jp/entry/kaggle-feature-management
[3]:https://github.com/flowlight0/talkingdata-adtracking-fraud-detection
[4]:https://github.com/upura/ml-competition-template-titanic
