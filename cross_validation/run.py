import json
import sys

import numpy as np
import pandas as pd

from db import table_load, table_write
from models import CatboostClassifier, LightGBM, XGBoostClassifier
from postprocessing import postprocessing
from utils import timer

if __name__ == "__main__":
    if len(sys.argv) == 2:
        config_file_name = sys.argv[1]
    else:
        config_file_name = "lightgbm_0"

    print("Config file Name: ", config_file_name)

    accuracies = []

    config: dict = json.load(open("./configs/{}.json".format(config_file_name)))

    models: dict = {
        "lightgbm": LightGBM,
        "xgbClassifier": XGBoostClassifier,
        "catboostClassifier": CatboostClassifier,
    }
    model = models[config["model"]["name"]]()

    n_splits = config["cv"]["n_splits"]
    random_state = config["cv"]["random_state"]
    target_cols = config["features"]["target"]
    train_cols = config["features"]["train"]
    categorical_cols = config["features"]["categorical"]
    params = config["model"]

    for n_fold in range(n_splits):
        with timer("CV No.{}".format(n_fold)):
            train = table_load("cv_train_{}".format(n_fold))
            valid = table_load("cv_test_{}".format(n_fold))
            y_real = valid[target_cols].iloc[:, 0].values.flatten()

            cv_model, y_pred = model.train_and_predict(
                train=train,
                valid=valid,
                weight=None,
                categorical_features=categorical_cols,
                target_cols=target_cols,
                train_cols=train_cols,
                params=params,
            )

            valid["survived"] = y_pred
            y_pred = postprocessing(train=train, test=valid)

            cv_result = pd.DataFrame(
                {
                    "index": valid.index,
                    "predicted": y_pred.flatten(),
                    "real": y_real,
                    "difference": y_pred.flatten() - y_real,
                    "difference_abs": abs(y_pred.flatten() - y_real),
                }
            )
            table_write(table_name="cv_result_{}".format(n_fold), df=cv_result)

            predicted = (y_pred.flatten() > 0.5).astype(int)
            accuracy = (predicted == y_real).sum() / len(predicted)
            accuracies.append(accuracy)

            print("Accuracy: {}".format(accuracy))

    print("Total Accuracy: {}".format(np.mean(accuracies)))
