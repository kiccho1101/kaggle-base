import numpy as np
import pandas as pd


def postprocessing(train: pd.DataFrame, test: pd.DataFrame):

    # if family survival rate is very high/low, we apply that value
    train.loc[:, "family"] = (
        train["family_name"].astype(str) + "___" + train["family_size"].astype(str)
    )
    test.loc[:, "family"] = (
        test["family_name"].astype(str) + "___" + test["family_size"].astype(str)
    )
    test.loc[test["family"] == "Andersson___6", "survived"] = 0
    test.loc[test["family"] == "Sage___10", "survived"] = 0
    test.loc[test["family"] == "Goodwin___7", "survived"] = 0
    test.loc[test["family"] == "Rice___5", "survived"] = 0
    test.loc[test["family"] == "Asplund___6", "survived"] = 1
    test.loc[test["family"] == "Palsson___4", "survived"] = 0
    test.loc[test["family"] == "Ford___4", "survived"] = 0
    test.loc[test["family"] == "Lefebre___4", "survived"] = 0
    test.loc[test["family"] == "Johansson___0", "survived"] = 0
    test.loc[test["family"] == "Daly___0", "survived"] = 1
    test.loc[test["family"] == "Danbom___2", "survived"] = 0
    test.loc[test["family"] == "Johnston___3", "survived"] = 0
    test.loc[test["family"] == "Coutts___2", "survived"] = 1
    test.loc[test["family"] == "Nakid___2", "survived"] = 1
    test.loc[test["family"] == "Sandstrom___2", "survived"] = 1
    test.loc[test["family"] == "Moubarek___2", "survived"] = 1
    test.loc[test["family"] == "Carlsson___0", "survived"] = 0
    test.loc[test["family"] == "Peter___2", "survived"] = 1
    test.loc[test["family"] == "Quick___2", "survived"] = 1
    test.loc[test["family"] == "Herman___3", "survived"] = 1
    test.loc[test["family"] == "Oreskovic___0", "survived"] = 0
    test.loc[test["family"] == "Rosblom___2", "survived"] = 0
    test.loc[test["family"] == "Olsson___0", "survived"] = 0
    test.loc[test["family"] == "Elias___2", "survived"] = 0
    test.loc[test["family"] == "Ryerson___4", "survived"] = 1
    test.loc[test["family"] == "Svensson___0", "survived"] = 0
    test.loc[test["family"] == "Boulos___2", "survived"] = 0
    test.loc[test["family"] == "McCoy___2", "survived"] = 1
    test.loc[test["family"] == "Becker___3", "survived"] = 1
    test.loc[test["family"] == "Wick___2", "survived"] = 1
    test.loc[test["family"] == "Cacic___0", "survived"] = 0
    test.loc[test["family"] == "Caldwell___2", "survived"] = 1

    # Rev of train data all died (6 Revs are in train data)
    test.loc[test["name_titles"] == "Rev", "survived"] = 0

    return test["survived"].values.flatten()
