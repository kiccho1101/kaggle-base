import json
import sys

from sklearn.model_selection import StratifiedKFold

from db import table_load, table_write
from utils import timer

if __name__ == "__main__":

    if len(sys.argv) == 2:
        config_file_name = sys.argv[1]
    else:
        config_file_name = "lightgbm_0"

    print("Config file Name: ", config_file_name)

    with timer("kfold"):
        config: dict = json.load(open("./configs/{}.json".format(config_file_name)))

        train = table_load("train")

        folds = StratifiedKFold(
            n_splits=config["cv"]["n_splits"],
            shuffle=True,
            random_state=config["cv"]["random_state"],
        ).split(train, train[config["features"]["target"]])

        for n_fold, (train_index, valid_index) in enumerate(folds):
            cv_train_df = train.loc[train_index]
            cv_test_df = train.loc[valid_index]
            table_write(table_name="cv_train_{}".format(n_fold), df=cv_train_df)
            table_write(table_name="cv_test_{}".format(n_fold), df=cv_test_df)
