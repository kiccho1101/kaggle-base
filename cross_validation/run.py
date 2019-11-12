import json
import sys

from models import LightGBM, Model
from utils import timer

models: dict = {"lightgbm": LightGBM}

if __name__ == "__main__":

    if len(sys.argv) == 2:
        config_file_name = sys.argv[1]
    else:
        config_file_name = "lightgbm_0"

    print("Config file Name: ", config_file_name)

    # Parse args
    config: dict = json.load(open("./configs/{}.json".format(config_file_name)))
    model: Model = models[config["model"]["name"]]()

    # Cross Validation
    cv_models, cv_result = model.cross_validatoin(
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
        target_cols=config["features"]["target"],
        train_cols=config["features"]["train"],
        categorical_cols=config["features"]["categorical"],
        params=config["model"],
    )
