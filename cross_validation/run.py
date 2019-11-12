import json
import sys

import numpy as np
import pandas as pd

from db import table_load
from models import LightGBM
from utils import timer

if __name__ == "__main__":
    if len(sys.argv) == 2:
        config_file_name = sys.argv[1]
    else:
        config_file_name = "lightgbm_0"

    print("Config file Name: ", config_file_name)

    config: dict = json.load(open("./configs/{}.json".format(config_file_name)))

    models: dict = {"lightgbm": LightGBM}
    model = models[config["model"]["name"]]()

    cv_result = pd.DataFrame()
    cv_models = []
    n_splits = config["cv"]["n_splits"]
    random_state = config["cv"]["random_state"]
    target_cols = config["features"]["target"]
    train_cols = config["features"]["train"]
    categorical_cols = config["features"]["categorical"]
    params = config["model"]

    for n_fold in range(n_splits):
        with timer("CV No.{}".format(n_fold)):
            train = table_load("cv_train_{}".format(n_fold), train_cols + target_cols)
            valid = table_load("cv_test_{}".format(n_fold), train_cols + target_cols)

            cv_model, y_pred = model.train_and_predict(
                train=train,
                valid=valid,
                weight=None,
                categorical_features=categorical_cols,
                target_cols=target_cols,
                train_cols=train_cols,
                params=params,
            )

            cv_models.append(cv_model)

            y_real = valid[target_cols].iloc[:, 0].values.flatten()
            cv_result = pd.concat(
                [
                    cv_result,
                    pd.DataFrame(
                        {
                            "index": valid.index,
                            "predicted": y_pred.flatten(),
                            "real": y_real,
                            "difference": y_pred.flatten() - y_real,
                            "n_fold": n_fold,
                        }
                    ),
                ]
            )

            predicted = (y_pred.flatten() > 0.5).astype(int)
            accuracy = (predicted == y_real).sum() / len(predicted)

            print("Accuracy: {}".format(accuracy))
