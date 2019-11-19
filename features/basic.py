import numpy as np
import pandas as pd

from features import Feature


class PclassOhe(Feature):
    def depends_on(self):
        return ["pclass"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        combined = pd.concat(
            [combined, pd.get_dummies(combined["pclass"], prefix="pclass")], axis=1
        )
        train, test = combined.iloc[: len(train)], combined[len(train) :]

        memo = self.create_memo(
            memo, "pclass_ohe", "binary", "pclass ohe", " ".join(self.depends_on()),
        )
        return train, test, memo


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


class NameTitlesInt(Feature):
    def depends_on(self):
        return ["name_titles"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        name_titles_mapper = (
            combined["name_titles"]
            .value_counts()
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
        )
        combined["name_titles_int"] = combined["name_titles"].map(name_titles_mapper)

        train, test = combined.iloc[: len(train)], combined[len(train) :]

        memo = self.create_memo(
            memo,
            "name_titles_int",
            "int",
            "Ordinal encoding of name_titles",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class NameTitlesOhe(Feature):
    def depends_on(self):
        return ["name_titles"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        combined = pd.concat(
            [combined, pd.get_dummies(combined["name_titles"], prefix="name_titles")],
            axis=1,
        )
        train, test = combined.iloc[: len(train)], combined[len(train) :]

        memo = self.create_memo(
            memo,
            "name_titles_ohe",
            "str",
            "Ohe of name_titles",
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


class NameTitlesSummarizedInt(Feature):
    def depends_on(self):
        return ["name_titles_summarized"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        name_titles_mapper = (
            combined["name_titles_summarized"]
            .value_counts()
            .reset_index()
            .reset_index()
            .set_index("index")["level_0"]
        )
        combined["name_titles_summarized_int"] = combined["name_titles_summarized"].map(
            name_titles_mapper
        )

        train, test = combined.iloc[: len(train)], combined[len(train) :]

        memo = self.create_memo(
            memo,
            "name_titles_summarized_int",
            "int",
            "Ordinal encoding of name_titles_summarized",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class AgeIsNull(Feature):
    def depends_on(self):
        return ["age"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        combined["age_is_null"] = combined["age"].isnull().astype(int)
        train, test = combined.iloc[: len(train)], combined.iloc[len(train) :]

        memo = self.create_memo(
            memo,
            "age_is_null",
            "binary",
            "age is NULL or not",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class CabinIsNull(Feature):
    def depends_on(self):
        return ["cabin"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        combined["cabin_is_null"] = combined["cabin"].isnull().astype(int)
        train, test = combined.iloc[: len(train)], combined.iloc[len(train) :]

        memo = self.create_memo(
            memo,
            "cabin_is_null",
            "binary",
            "cabin is NULL or not",
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


class FamilyName(Feature):
    def depends_on(self):
        return ["name"]

    def create_features(self, train, test, memo):
        train["family_name"] = train["name"].str.split(",", expand=True).iloc[:, 0]
        test["family_name"] = test["name"].str.split(",", expand=True).iloc[:, 0]

        memo = self.create_memo(
            memo, "family_name", "str", "Family Name", " ".join(self.depends_on()),
        )
        return train, test, memo


class FamilyTargetEncodingMean(Feature):
    def depends_on(self):
        return ["family_name", "family_size", "pclass_target_encoding_mean", "survived"]

    def create_features(self, train, test, memo):
        train["family"] = train["family_name"] + train["family_size"].astype(str)
        test["family"] = test["family_name"] + test["family_size"].astype(str)
        family_survived_rate = train.groupby("family")["survived"].mean()
        train["family_target_encoding_mean"] = train["family"].map(family_survived_rate)
        test["family_target_encoding_mean"] = test["family"].map(family_survived_rate)

        memo = self.create_memo(
            memo,
            "family_target_encoding_mean",
            "float",
            "Survive rate of family",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class TicketStartswith(Feature):
    def depends_on(self):
        return ["ticket"]

    def create_features(self, train, test, memo):
        train["ticket_startswith"] = train["ticket"].str[:1]
        test["ticket_startswith"] = test["ticket"].str[:1]

        memo = self.create_memo(
            memo,
            "ticket_startswith",
            "str",
            "Start string of ticket",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class TicketStartswithTargetEncodingMean(Feature):
    def depends_on(self):
        return ["ticket_startswith", "survived"]

    def create_features(self, train, test, memo):
        survival_rate = train.groupby("ticket_startswith")["survived"].mean()
        train["ticket_startswith_target_encoding_mean"] = (
            train["ticket_startswith"].map(survival_rate).astype(float)
        )
        test["ticket_startswith_target_encoding_mean"] = (
            test["ticket_startswith"].map(survival_rate).astype(float)
        )

        memo = self.create_memo(
            memo,
            "ticket_startswith_target_encoding_mean",
            "float",
            "Survival rate of same ticket start string",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class FamilyTargetEncodingMeanFilled(Feature):
    def depends_on(self):
        return ["family_target_encoding_mean", "ticket_startswith_target_encoding_mean"]

    def create_features(self, train, test, memo):
        train["family_target_encoding_mean_filled"] = (
            train["family_target_encoding_mean"]
            .fillna(train["ticket_startswith_target_encoding_mean"])
            .astype(float)
        )
        test["family_target_encoding_mean_filled"] = (
            test["family_target_encoding_mean"]
            .fillna(test["ticket_startswith_target_encoding_mean"])
            .astype(float)
        )

        memo = self.create_memo(
            memo,
            "family_target_encoding_mean_filled",
            "float",
            "Family survival rate filled with ticket TE",
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


class EmbarkedOhe(Feature):
    def depends_on(self):
        return ["embarked"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)
        combined = pd.concat(
            [combined, pd.get_dummies(combined["embarked"], prefix="embarked")], axis=1
        )
        train, test = combined.iloc[: len(train)], combined[len(train) :]

        memo = self.create_memo(
            memo, "embarked_ohe", "binary", "embarked ohe", " ".join(self.depends_on()),
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


class FareCut(Feature):
    def depends_on(self):
        return ["fare_filled"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)

        fare_cut = pd.cut(combined["fare_filled"], 5).astype(str)
        train["fare_cut"] = fare_cut.iloc[: len(train)]
        test["fare_cut"] = fare_cut.iloc[len(train) :]

        fare_cut_onehot = pd.get_dummies(
            pd.cut(combined["fare_filled"], 5).astype(str), prefix="fare_cut",
        )
        train = pd.concat([train, fare_cut_onehot.iloc[: len(train)]], axis=1)
        test = pd.concat([test, fare_cut_onehot.iloc[len(train) :]], axis=1)

        rename_cols = {
            col_name: "fare_cut_{}".format(i)
            for i, col_name in enumerate(train.filter(like="fare_cut_").columns)
        }
        train = train.rename(columns=rename_cols)
        test = test.rename(columns=rename_cols)

        memo = self.create_memo(
            memo,
            "fare_cut",
            "str",
            "fare cut by pd.cut into 5",
            " ".join(self.depends_on()),
        )
        return train, test, memo


class FareQcut(Feature):
    def depends_on(self):
        return ["fare_filled"]

    def create_features(self, train, test, memo):
        combined = pd.concat([train, test], axis=0, sort=True)

        fare_cut = pd.qcut(combined["fare_filled"], 5).astype(str)
        train["fare_qcut"] = fare_cut.iloc[: len(train)]
        test["fare_qcut"] = fare_cut.iloc[len(train) :]

        fare_cut_onehot = pd.get_dummies(
            pd.cut(combined["fare_filled"], 5).astype(str), prefix="fare_qcut",
        )
        train = pd.concat([train, fare_cut_onehot.iloc[: len(train)]], axis=1)
        test = pd.concat([test, fare_cut_onehot.iloc[len(train) :]], axis=1)

        rename_cols = {
            col_name: "fare_qcut_{}".format(i)
            for i, col_name in enumerate(train.filter(like="fare_qcut_").columns)
        }
        train = train.rename(columns=rename_cols)
        test = test.rename(columns=rename_cols)

        memo = self.create_memo(
            memo,
            "fare_qcut",
            "str",
            "fare cut by pd.qcut into 5",
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
