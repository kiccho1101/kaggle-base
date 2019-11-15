import inspect
import sys

from utils import timer

from features.fill_null import *  # isort:skip
from features.basic import *  # isort:skip
from features.categorical import *  # isort:skip


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

    with timer("Create features"):
        features = get_features()
        if len(sys.argv[1:]) == 0:
            exec_feature_list = features
        else:
            exec_feature_list = sys.argv[1:]

        for feature in exec_feature_list:
            features[feature]().run()
