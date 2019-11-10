import copy
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Booster

from models.base import Model


class LightGBM(Model):
    def train_and_predict(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        weight,
        categorical_features: List[str],
        target_cols: List[str],
        train_cols: List[str],
        params: dict,
    ) -> (Booster, pd.DataFrame):

        if weight is None:
            d_train = lgb.Dataset(
                train[train_cols], label=train[target_cols].iloc[:, 0],
            )
        else:
            d_train = lgb.Dataset(
                train[train_cols], label=train[target_cols].iloc[:, 0], weight=weight,
            )

        model = lgb.train(params["model_params"], d_train, **params["train_params"])

        y_pred = model.predict(
            valid[train_cols],
            categorical_feature=categorical_features,
            num_iteration=model.best_iteration,
        )
        return model, y_pred

    def metric(self, y_pred: np.array, y_real: np.array):
        predicted = (y_pred > 0.5).astype(int)
        accuracy = (predicted == y_real).sum() / len(predicted)
        return accuracy
