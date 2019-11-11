import argparse
import json

from sklearn.model_selection import StratifiedKFold

from db import table_load, table_write

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lightgbm_0")
    options = parser.parse_args()
    config: dict = json.load(open("./configs/{}.json".format(options.config)))

    train = table_load("train")

    folds = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=True,
        random_state=config["cv"]["random_state"],
    ).split(train, train[config["features"]["target"]])

    for n_fold, (train_index, valid_index) in enumerate(folds):
        cv_train_df = train.loc[train_index]
        cv_test_df = train.loc[valid_index]
        table_write(df=cv_train_df, table_name="cv_train_{}".format(n_fold))
        table_write(df=cv_test_df, table_name="cv_test_{}".format(n_fold))
