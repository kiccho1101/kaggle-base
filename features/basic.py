import numpy as np
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

        train["is_mr"] = (train["name_titles"] == "Mr").astype(int)
        test["is_mr"] = (test["name_titles"] == "Mr").astype(int)

        train["is_miss"] = (train["name_titles"] == "Miss").astype(int)
        test["is_miss"] = (test["name_titles"] == "Miss").astype(int)

        train["is_mrs"] = (train["name_titles"] == "Mrs").astype(int)
        test["is_mrs"] = (test["name_titles"] == "Mrs").astype(int)

        train["is_master"] = (train["name_titles"] == "Master").astype(int)
        test["is_master"] = (test["name_titles"] == "Master").astype(int)

        memo = self.create_memo(
            memo,
            "name_titles",
            "str",
            "Mr, Mrs, Miss, etc...",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class NameTitlesSummarized(Feature):
    def depends_on(self):
        return ["name_titles"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        combined["name_titles_summarized"] = combined["name_titles"]
        combined.loc[
            ~combined["name_titles"].str.match(r"Mrs|Mr|Miss|Master|Dr|Rev"),
            "name_titles_summarized",
        ] = "other"
        train, test = combined.iloc[: len(train)], combined.iloc[len(train) :]

        memo = self.create_memo(
            memo,
            "name_titles_summarized",
            "str",
            "Mr, Mrs, Miss, ... Other",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class AgeFilled(Feature):
    def depends_on(self):
        return ["age", "pclass", "sex", "name_titles_summarized"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        age_map = combined.groupby(["pclass", "sex", "name_titles_summarized"])[
            "age"
        ].median()
        combined["age_filled"] = combined["age"].fillna(
            combined.apply(
                lambda x: age_map[x["pclass"]][x["sex"]][x["name_titles_summarized"]],
                axis=1,
            )
        )
        combined["age_filled"] = (
            combined["age_filled"].fillna(combined["age"].median()).astype(int)
        )
        train, test = combined.iloc[: len(train)], combined.iloc[len(train) :]

        memo = self.create_memo(
            memo, "age_filled", "float", "age filled", " ".join(self.depends_on()),
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


class EmbarkedIsC(Feature):
    def depends_on(self):
        return ["embarked"]

    def create_features(self, train, test, memo):
        train["embarked_is_c"] = (train["embarked"] == "C").astype(int)
        test["embarked_is_c"] = (test["embarked"] == "C").astype(int)

        memo = self.create_memo(
            memo,
            "embarked_is_c",
            "binary",
            "Embarked from c or not",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class AgeCut(Feature):
    def depends_on(self):
        return ["age_filled"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)

        age_cut = pd.cut(combined["age_filled"], 5).astype(str)
        train["age_cut"] = age_cut.iloc[: len(train)]
        test["age_cut"] = age_cut.iloc[len(train) :]

        age_cut_onehot = pd.get_dummies(
            pd.cut(combined["age_filled"], 5).astype(str), prefix="age_cut",
        )
        train = pd.concat([train, age_cut_onehot.iloc[: len(train)]], axis=1)
        test = pd.concat([test, age_cut_onehot.iloc[len(train) :]], axis=1)

        rename_cols = {
            col_name: "age_cut_{}".format(i)
            for i, col_name in enumerate(train.filter(like="age_cut_").columns)
        }
        train = train.rename(columns=rename_cols)
        test = test.rename(columns=rename_cols)

        memo = self.create_memo(
            memo,
            "age_cut",
            "str",
            "age cut by pd.cut into 5",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class FareLog(Feature):
    def depends_on(self):
        return ["fare"]

    def create_features(self, train, test, memo):
        train["fare_log"] = np.log(train["fare"] + 0.0001)
        test["fare_log"] = np.log(test["fare"] + 0.0001)

        memo = self.create_memo(
            memo, "fare_log", "float", "Log fare", " ".join(self.depends_on()),
        )
        return train, test, memo
