import copy
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class CatboostClassifier:
    def train_and_predict(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        weight,
        categorical_features: List[str],
        target_cols: List[str],
        train_cols: List[str],
        params: dict,
    ):

        model = CatBoostClassifier(**params["model_params"])
        model.fit(train[train_cols], train[target_cols].iloc[:, 0], verbose=False)

        y_pred = model.predict(valid[train_cols])
        return model, y_pred
