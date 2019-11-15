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


class IsMr(Feature):
    def depends_on(self):
        return ["name_titles"]

    def create_features(self, train, test, memo):
        train["is_mr"] = (train["name_titles"] == "Mr").astype(int)
        test["is_mr"] = (test["name_titles"] == "Mr").astype(int)

        memo = self.create_memo(
            memo, "is_mr", "binary", "Is Mr or not", " ".join(self.depends_on()),
        )
        return train, test, memo


class IsMiss(Feature):
    def depends_on(self):
        return ["name_titles"]

    def create_features(self, train, test, memo):
        train["is_miss"] = (train["name_titles"] == "Miss").astype(int)
        test["is_miss"] = (test["name_titles"] == "Miss").astype(int)

        memo = self.create_memo(
            memo, "is_miss", "binary", "Is Miss or not", " ".join(self.depends_on()),
        )
        return train, test, memo


class IsMrs(Feature):
    def depends_on(self):
        return ["name_titles"]

    def create_features(self, train, test, memo):
        train["is_mrs"] = (train["name_titles"] == "Mrs").astype(int)
        test["is_mrs"] = (test["name_titles"] == "Mrs").astype(int)

        memo = self.create_memo(
            memo, "is_mrs", "binary", "Is Mrs or not", " ".join(self.depends_on()),
        )
        return train, test, memo


class IsMaster(Feature):
    def depends_on(self):
        return ["name_titles"]

    def create_features(self, train, test, memo):
        train["is_master"] = (train["name_titles"] == "Master").astype(int)
        test["is_master"] = (test["name_titles"] == "Master").astype(int)

        memo = self.create_memo(
            memo,
            "is_master",
            "binary",
            "Is Master or not",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class FamilySize(Feature):
    def depends_on(self):
        return ["sib_sp", "parch"]

    def create_features(self, train, test, memo):
        train["family_size"] = (train["sib_sp"] + train["parch"]).astype(int)
        test["family_size"] = (test["sib_sp"] + test["parch"]).astype(int)

        memo = self.create_memo(
            memo,
            "family_size",
            "int",
            "# of family on board",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class IsAlone(Feature):
    def depends_on(self):
        return ["family_size"]

    def create_features(self, train, test, memo):
        train["is_alone"] = (train["family_size"] == 0).astype(int)
        test["is_alone"] = (test["family_size"] == 0).astype(int)

        memo = self.create_memo(
            memo,
            "is_alone",
            "binary",
            "No family on board",
            " ".join(self.depends_on()),
        )
        return train, test, memo
