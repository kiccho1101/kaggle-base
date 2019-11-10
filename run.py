import argparse
import json

from db import table_load
from models import LightGBM, Model

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
    cv_models, result = model.cross_validatoin(
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
        train=train,
        target_cols=config["features"]["target"],
        train_cols=config["features"]["train"],
        categorical_cols=config["features"]["categorical"],
        params=config["model"],
    )
