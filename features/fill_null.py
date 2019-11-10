import pandas as pd

from features import Feature


class AgeNullFilledWithMode(Feature):
    def create_features(self):
        combined = pd.concat([self.train, self.test], axis=0, sort=True)
        self.train[self.name] = (
            self.train["age"]
            .fillna(int(combined["age"].value_counts().index[0]))
            .astype(int)
        )
        self.test[self.name] = (
            self.train["age"]
            .fillna(int(combined["age"].value_counts().index[0]))
            .astype(int)
        )

        self.create_memo(self.name, "int", "age NaN-filled with mode", "age")


class EmbarkedNullFilledWithMode(Feature):
    def create_features(self):
        combined = pd.concat([self.train, self.test], axis=0, sort=True)
        self.train[self.name] = self.train["embarked"].fillna(
            combined["embarked"].value_counts().index[0]
        )
        self.test[self.name] = self.train["embarked"].fillna(
            combined["embarked"].value_counts().index[0]
        )

        self.create_memo(self.name, "str", "embarked NaN-filled with mode", "embarked")
