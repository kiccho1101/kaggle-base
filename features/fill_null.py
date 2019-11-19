import pandas as pd

from features import Feature


class AgeNullFilledWithMode(Feature):
    def depends_on(self):
        return ["age"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        train["age_null_filled_with_mode"] = (
            train["age"]
            .fillna(int(combined["age"].value_counts().index[0]))
            .astype(int)
        )
        test["age_null_filled_with_mode"] = (
            test["age"].fillna(int(combined["age"].value_counts().index[0])).astype(int)
        )

        memo = self.create_memo(
            memo,
            "age_null_filled_with_mode",
            "int",
            "age NaN-filled with mode",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class EmbarkedNullFilledWithMode(Feature):
    def depends_on(self):
        return ["embarked"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        train["embarked_null_filled_with_mode"] = train["embarked"].fillna(
            combined["embarked"].value_counts().index[0]
        )
        test["embarked_null_filled_with_mode"] = test["embarked"].fillna(
            combined["embarked"].value_counts().index[0]
        )

        memo = self.create_memo(
            memo,
            "embarked_null_filled_with_mode",
            "str",
            "embarked NaN-filled with mode",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class FareFilled(Feature):
    def depends_on(self):
        return ["fare", "pclass", "sex"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        fare_map = combined.groupby(["pclass", "sex"])["fare"].median()

        combined["fare_filled"] = combined["fare"].fillna(
            combined.apply(lambda x: fare_map[x["pclass"]][x["sex"]], axis=1)
        )

        train, test = combined.iloc[: len(train)], combined.iloc[len(train) :]

        memo = self.create_memo(
            memo,
            "fare_filled",
            "float",
            "fare filled with median values grouped by (pclass, sex)",
            " ".join(self.depends_on()),
        )
        return train, test, memo
