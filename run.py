import argparse
import json

from db import table_load
from models import LightGBM, Model
from submit import create_submission_file

models: dict = {"lightgbm": LightGBM}


if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="lightgbm_0")
    parser.add_argument("--train_only", default=False, action="store_true")
    options = parser.parse_args()
    config: dict = json.load(open("./configs/{}.json".format(options.config)))

    model: Model = models[config["model"]["name"]]()

    # Load data
    train = table_load(
        table_name=config["dataset"]["train_table"],
        cols=config["features"]["id"]
        + config["features"]["target"]
        + config["features"]["train"],
    )
    test = table_load(
        table_name=config["dataset"]["test_table"],
        cols=config["features"]["id"] + config["features"]["train"],
    )

    # Cross Validation
    cv_models, cv_result = model.cross_validatoin(
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
        train=train,
        target_cols=config["features"]["target"],
        train_cols=config["features"]["train"],
        categorical_cols=config["features"]["categorical"],
        params=config["model"],
    )

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

    # Create submission file
    create_submission_file(
        PassengerId=test[config["features"]["id"]].iloc[:, 0].tolist(),
        Survived=(y_pred > 0.5).astype(int).tolist(),
        exp_name=options.config,
    )
