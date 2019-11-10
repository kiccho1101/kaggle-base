import re
from abc import ABCMeta, abstractmethod

import pandas as pd

from utils import timer


class Feature(metaclass=ABCMeta):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, memo: pd.DataFrame):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            # Rename to snake_case
            self.name = re.sub(
                "([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__
            ).lstrip("_")

        self.train = train
        self.test = test
        self.memo = memo

    def create_memo(
        self, feature_name: str, data_type: str, description: str, depends_on: str
    ):
        self.memo = (
            pd.concat(
                [
                    self.memo,
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

    def run(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        with timer(self.name):
            self.create_features()
        return self.train, self.test, self.memo

    @abstractmethod
    def create_features(self):
        raise NotImplementedError
