import json
import sys

from db import table_load
from models import CatboostClassifier, LightGBM, XGBoostClassifier
from postprocessing import postprocessing
from submit import create_submission_file
from utils import timer

models: dict = {
    "lightgbm": LightGBM,
    "xgbClassifier": XGBoostClassifier,
    "catboostClassifier": CatboostClassifier,
}

if __name__ == "__main__":

    if len(sys.argv) == 2:
        config_file_name = sys.argv[1]
    else:
        config_file_name = "lightgbm_0"

    print("Config file Name: ", config_file_name)

    # Parse args
    config: dict = json.load(open("./configs/{}.json".format(config_file_name)))
    model = models[config["model"]["name"]]()

    with timer("train and predict"):

        # Load data
        train = table_load(table_name=config["dataset"]["train_table"])
        test = table_load(table_name=config["dataset"]["test_table"])

        # Train and predict
        result_model, y_pred = model.train_and_predict(
            train=train,
            valid=test,
            weight=None,
            categorical_features=config["features"]["categorical"],
            target_cols=config["features"]["target"],
            train_cols=config["features"]["train"],
            params=config["model"],
        )

        test["survived"] = y_pred
        y_pred = postprocessing(train=train, test=test)

        # Create submission file
        create_submission_file(
            PassengerId=test[config["features"]["id"]].iloc[:, 0].tolist(),
            Survived=(y_pred > 0.5).astype(int).tolist(),
            exp_name=config_file_name,
        )
