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
