from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.model_selection import StratifiedKFold

from db import table_load
from utils import timer


class Model:
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def metric(self, y_pred: np.array, y_real: np.array):
        raise NotImplementedError

    def cross_validatoin(
        self,
        n_splits: int,
        random_state: int,
        target_cols: List[str],
        train_cols: List[str],
        categorical_cols: List[str],
        params: dict,
    ) -> (List[Booster], pd.DataFrame):

        result = pd.DataFrame()
        cv_models = []

        for n_fold in range(n_splits):
            with timer("CV No.{}".format(n_fold)):
                train = table_load(
                    "cv_train_{}".format(n_fold), train_cols + target_cols
                )
                valid = table_load(
                    "cv_test_{}".format(n_fold), train_cols + target_cols
                )

                model, y_pred = self.train_and_predict(
                    train=train,
                    valid=valid,
                    weight=None,
                    categorical_features=categorical_cols,
                    target_cols=target_cols,
                    train_cols=train_cols,
                    params=params,
                )

                cv_models.append(model)

                y_real = valid[target_cols].iloc[:, 0].values.flatten()
                result = pd.concat(
                    [
                        result,
                        pd.DataFrame(
                            {
                                "index": valid.index,
                                "predicted": y_pred.flatten(),
                                "real": y_real,
                                "difference": y_pred.flatten() - y_real,
                                "n_fold": n_fold,
                            }
                        ),
                    ]
                )

                accuracy = self.metric(y_pred=y_pred.flatten(), y_real=y_real)
                print("Accuracy: {}".format(accuracy))

        return cv_models, result
