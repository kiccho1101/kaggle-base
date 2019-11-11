import re
from abc import ABCMeta, abstractmethod

import pandas as pd

from db import table_load, table_write, insert_cols
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

        self.train = table_load(table_name="train", cols=self.depends_on())
        self.test = table_load(table_name="test", cols=self.depends_on())
        self.memo = table_load(table_name="memo")

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
        table_write(self.memo, table_name="memo")

    def run(self):
        with timer(self.name):
            self.create_features()
            insert_cols(table_name="train", df=self.train)
            insert_cols(table_name="test", df=self.test)

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    @abstractmethod
    def depends_on(self):
        raise NotImplementedError
