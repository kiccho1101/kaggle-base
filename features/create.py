import inspect
import sys

from db import table_load, table_write
from features.fill_null import *


def get_features() -> Dict[str, Any]:
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

    features = get_features()
    exec_feature_list = sys.argv[1:]

    if len(exec_feature_list) == 0:
        for feature in features:
            train, test, memo = features[feature](train, test, memo).run()

    else:
        for feature in exec_feature_list:
            train, test, memo = features[feature](train, test, memo).run()

    print()
    table_write(train, "train")
    table_write(test, "test")
    table_write(memo, "memo")
    memo.to_csv("input/memo.csv")
    print()
