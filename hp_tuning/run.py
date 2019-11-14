import json
import logging
import pickle
import sys

import lightgbm as lgb
import numpy as np
import optuna
import sklearn.metrics
from sklearn.model_selection import train_test_split

from db import table_load


def lgb_objective(trial):
    train = table_load(table_name=config["dataset"]["train_table"])[
        config["features"]["train"] + config["features"]["target"]
    ]
    train_x, test_x, train_y, test_y = train_test_split(
        train[config["features"]["train"]],
        train[config["features"]["target"]].iloc[:, 0],
        test_size=0.25,
    )

    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 10.0),
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(test_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
    return accuracy


if __name__ == "__main__":

    if len(sys.argv) == 2:
        config_file_name = sys.argv[1]
    else:
        config_file_name = "lightgbm_0"

    print("Config file Name: ", config_file_name)

    config: dict = json.load(open("./configs/{}.json".format(config_file_name)))

    if config["model"]["name"] == "lightgbm":
        objective = lgb_objective

    log_file_name = "./output/hp_tuning_results/{}_log.txt".format(config_file_name)
    with open(log_file_name, "w"):
        pass

    study = optuna.create_study(direction="maximize")
    logging.getLogger().setLevel(logging.INFO)  # Setup the root logger.
    logging.getLogger().addHandler(logging.FileHandler(log_file_name))

    optuna.logging.enable_propagation()
    study.optimize(objective, n_trials=100)

    logging.getLogger().info("Best trial:")
    trial = study.best_trial

    logging.getLogger().info("  Value: {}".format(trial.value))

    logging.getLogger().info("  Params: ")
    for key, value in trial.params.items():
        logging.getLogger().info("    {}: {}".format(key, value))

    with open(
        "./output/hp_tuning_results/{}_study.pickle".format(config_file_name), "wb"
    ) as f:
        pickle.dump(study, f)
