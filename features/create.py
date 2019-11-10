import inspect
import sys

from db import table_load, table_write
from features.fill_null import *
from utils import timer


def get_features() -> dict:
    features = {}
    namespace = globals()
    for var_name in list(namespace):
        variable = namespace[var_name]
        if (
            inspect.isclass(variable)
            and issubclass(variable, Feature)
            and not inspect.isabstract(variable)
        ):
            features[var_name] = variable
    return features


if __name__ == "__main__":
    print()
    train = table_load("train")
    test = table_load("test")
    memo = table_load("memo")
    print()

    with timer("Create features"):
        features = get_features()
        if sys.argv[1:] == 0:
            exec_feature_list = features
        else:
            exec_feature_list = sys.argv[1:]

        for feature in exec_feature_list:
            train, test, memo = features[feature](train, test, memo).run()

    print()
    table_write(train, "train")
    table_write(test, "test")
    table_write(memo, "memo")
    memo.to_csv("input/memo.csv")
    print()
