import re
from abc import ABCMeta, abstractmethod

import pandas as pd

from db import find_table_name, insert_cols, table_load, table_write
from utils import timer


class Feature(metaclass=ABCMeta):
    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            # Rename to snake_case
            self.name = re.sub(
                "([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__
            ).lstrip("_")

    def create_memo(
        self,
        memo: pd.DataFrame,
        feature_name: str,
        data_type: str,
        description: str,
        depends_on: str,
    ):
        memo = (
            pd.concat(
                [
                    memo,
                    pd.DataFrame(
                        [
                            {
                                "feature": feature_name,
                                "data_type": data_type,
                                "description": description,
                                "depends_on": depends_on,
                            }
                        ]
                    ),
                ],
                axis=0,
                sort=True,
            )
            .drop_duplicates()
            .reset_index()
            .drop("index", axis=1)[
                ["feature", "data_type", "description", "depends_on"]
            ]
        )
        return memo

    def target_cols(self):
        return ["survived"]

    def run(self):
        with timer(self.name):
            train = table_load(table_name="train", cols=self.depends_on())
            test = table_load(
                table_name="test",
                cols=[
                    col for col in self.depends_on() if col not in self.target_cols()
                ],
            )
            memo = table_load(table_name="memo")
            train, test, memo = self.create_features(train, test, memo)
            insert_cols(table_name="train", df=train)
            insert_cols(table_name="test", df=test)
            table_write(table_name="memo", df=memo)

            cv_train_tables = find_table_name(like="cv_train", unlike="stats")[
                "table_name"
            ].tolist()
            cv_test_tables = find_table_name("cv_test", unlike="stats")[
                "table_name"
            ].tolist()
            if len(cv_train_tables) != len(cv_test_tables):
                raise ValueError("# of cv_train is not equal to # of cv_test!")
            for n_fold in range(len(cv_train_tables)):
                train = table_load(
                    table_name=cv_train_tables[n_fold], cols=self.depends_on()
                )
                test = table_load(
                    table_name=cv_test_tables[n_fold],
                    cols=[
                        col
                        for col in self.depends_on()
                        if col not in self.target_cols()
                    ],
                )
                train, test, memo = self.create_features(train, test, memo)
                insert_cols(table_name=cv_train_tables[n_fold], df=train)
                insert_cols(table_name=cv_test_tables[n_fold], df=test)

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    @abstractmethod
    def depends_on(self):
        raise NotImplementedError
