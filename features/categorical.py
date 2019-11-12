import pandas as pd

from features import Feature


class SexInt(Feature):
    def depends_on(self):
        return ["sex"]

    def create_features(self, train, test, memo):
        train["sex_int"] = train["sex"].map({"female": 0, "male": 1})
        test["sex_int"] = test["sex"].map({"female": 0, "male": 1})

        memo = self.create_memo(
            memo, "sex_int", "int", "sex encoded as int", " ".join(self.depends_on()),
        )
        return train, test, memo
