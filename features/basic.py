import pandas as pd

from features import Feature


class PclassFrequency(Feature):
    def depends_on(self):
        return ["pclass"]

    def create_features(self, train, test, memo):
        train["pclass_frequency"] = (
            train["pclass"].map(train["pclass"].value_counts()).astype(int)
        )
        test["pclass_frequency"] = (
            test["pclass"].map(test["pclass"].value_counts()).astype(int)
        )

        memo = self.create_memo(
            memo,
            "pcalss_frequency",
            "int",
            "High number means ordinary -> not survived",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class PclassTargetEncodingMean(Feature):
    def depends_on(self):
        return ["pclass", "survived"]

    def create_features(self, train, test, memo):
        train["pclass_target_encoding_mean"] = (
            train["pclass"]
            .map(train.groupby("pclass")["survived"].mean())
            .astype(float)
        )
        test["pclass_target_encoding_mean"] = (
            test["pclass"].map(train.groupby("pclass")["survived"].mean()).astype(float)
        )

        memo = self.create_memo(
            memo,
            "pclass_target_encoding_mean",
            "float",
            "High number means more survived",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class NameTitles(Feature):
    def depends_on(self):
        return ["name"]

    def create_features(self, train, test, memo):
        train["name_titles"] = train["name"].str.extract(r"([^\ ]*)\.").iloc[:, 0]
        test["name_titles"] = test["name"].str.extract(r"([^\ ]*)\.").iloc[:, 0]

        memo = self.create_memo(
            memo,
            "name_titles",
            "str",
            "Mr, Mrs, Miss, etc...",
            " ".join(self.depends_on()),
        )
        return train, test, memo
