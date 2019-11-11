import pandas as pd

from features import Feature
from utils import timer


class AgeNullFilledWithMode(Feature):
    def depends_on(self):
        return ["age"]

    def create_features(self):
        combined = pd.concat([self.train, self.test], axis=0, sort=True)
        self.train["age_null_filled_with_mode"] = (
            self.train["age"]
            .fillna(int(combined["age"].value_counts().index[0]))
            .astype(int)
        )
        self.test["age_null_filled_with_mode"] = (
            self.train["age"]
            .fillna(int(combined["age"].value_counts().index[0]))
            .astype(int)
        )

        self.create_memo(
            "age_null_filled_with_mode",
            "int",
            "age NaN-filled with mode",
            " ".join(self.depends_on()),
        )


class EmbarkedNullFilledWithMode(Feature):
    def depends_on(self):
        return ["embarked"]

    def create_features(self):
        combined = pd.concat([self.train, self.test], axis=0, sort=True)
        self.train["embarked_null_filled_with_mode"] = self.train[
            "embarked"
        ].fillna(combined["embarked"].value_counts().index[0])
        self.test["embarked_null_filled_with_mode"] = self.train["embarked"].fillna(
            combined["embarked"].value_counts().index[0]
        )

        self.create_memo(
            "embarked_null_filled_with_mode",
            "str",
            "embarked NaN-filled with mode",
            " ".join(self.depends_on()),
        )
