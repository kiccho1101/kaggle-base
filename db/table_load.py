import os

import pandas as pd
import psycopg2

from utils.__init__ import timer


def table_load(table_name="", cols=None):

    with timer("table load ({})".format(table_name)):

        # Drop table train, test
        with psycopg2.connect(
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=5432,
            database=os.environ["PROJECT_NAME"],
        ) as conn:

            if cols is None:
                df = pd.read_sql(
                    "SELECT * FROM {};".format(table_name), con=conn
                ).set_index("index")

            else:
                df = pd.read_sql(
                    "SELECT {} FROM {};".format(", ".join(cols), table_name), con=conn
                ).set_index("index")
            return df
