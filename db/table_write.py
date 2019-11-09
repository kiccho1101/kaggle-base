import os
import re

import pandas as pd
from sqlalchemy import create_engine

from utils.__init__ import timer


def table_write(df=pd.DataFrame(), table_name=""):

    with timer("table write ({})".format(table_name)):

        # Rename cols to snake-case
        df.columns = (
            pd.Series(df.columns)
            .map(
                lambda col: re.sub(
                    "([A-Z])", lambda x: "_" + x.group(1).lower(), col
                ).lstrip("_")
            )
            .to_list()
        )

        engine = create_engine(
            "postgresql://{}:{}@{}:5432/{}".format(
                os.environ["POSTGRES_USER"],
                os.environ["POSTGRES_PASSWORD"],
                os.environ["POSTGRES_HOST"],
                os.environ["PROJECT_NAME"],
            )
        )
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=True)
