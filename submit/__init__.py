import glob
import os

import pandas as pd

from utils import timer


def create_submission_file(PassengerId, Survived, exp_name: str):

    with timer("Create submission file"):
        submission_no = (
            len(
                glob.glob(
                    os.environ["PROJECT_DIR"]
                    + "/output/submission/submission_{}_{}_*.csv".format(
                        pd.to_datetime("today").strftime("%Y-%m-%d"), exp_name
                    )
                )
            )
            + 1
        )

        submission_file_name = "output/submission/submission_{}_{}_{}.csv".format(
            pd.to_datetime("today").strftime("%Y-%m-%d"), exp_name, submission_no
        )

        pd.DataFrame({"PassengerId": PassengerId, "Survived": Survived}).to_csv(
            os.environ["PROJECT_DIR"] + "/" + submission_file_name, index=False
        )

        print("Submit file is written: ", submission_file_name)
