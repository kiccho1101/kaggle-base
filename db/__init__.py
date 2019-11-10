import os
import re

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

from utils import timer


def exec_query(query: str):

    query = query.replace("'", "''")

    with psycopg2.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=5432,
        database=os.environ["PROJECT_NAME"],
    ) as conn, conn.cursor() as cur:
        cur.execute(query)
        conn.commit()


def table_load(table_name: str, cols=None) -> pd.DataFrame:

    with timer("table load ({})".format(table_name)):

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
                    "SELECT index, {} FROM {};".format(", ".join(cols), table_name),
                    con=conn,
                ).set_index("index")
            return df


def table_write(df: pd.DataFrame, table_name: str):

    with timer("table write ({})".format(table_name)):

        # Rename cols to snake_case
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
