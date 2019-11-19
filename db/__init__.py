import os
import re

import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

from utils import timer


def insert_cols(table_name: str, df: pd.DataFrame):
    # Rename cols to snake_case
    df.columns = (
        pd.Series(df.columns)
        .map(
            lambda col: re.sub(
                "([A-Z])", lambda x: "_" + x.group(1).lower(), col
            ).lstrip("_")
        )
        .tolist()
    )

    engine = create_engine(
        "postgresql://{}:{}@{}:5432/{}".format(
            os.environ["POSTGRES_USER"],
            os.environ["POSTGRES_PASSWORD"],
            os.environ["POSTGRES_HOST"],
            os.environ["PROJECT_NAME"],
        )
    )
    df.to_sql(name=table_name + "_tmp", con=engine, if_exists="replace", index=True)

    dtype_mapper = {
        np.dtype("object"): "TEXT",
        np.dtype("int32"): "BIGINT",
        np.dtype("int64"): "BIGINT",
        np.dtype("uint8"): "BIGINT",
        np.dtype("float64"): "DOUBLE PRECISION",
    }

    queries = [
        """
        ALTER TABLE {0} DROP COLUMN IF EXISTS {1};
        ALTER TABLE {0} ADD COLUMN IF NOT EXISTS {1} {2};
        UPDATE {0} t1
        SET    {1} = t2.{1}
        FROM   {0}_tmp t2
        WHERE  t1.index = t2.index
        """.format(
            table_name, col_name, dtype
        )
        for col_name, dtype in df.dtypes.map(dtype_mapper).iteritems()
    ]

    with psycopg2.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=5432,
        database=os.environ["PROJECT_NAME"],
    ) as conn, conn.cursor() as cur:
        for query in queries:
            cur.execute(query)
        conn.commit()

        cur.execute("DROP TABLE {}_tmp;".format(table_name))
        conn.commit()


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


def find_table_name(like: str, unlike=None) -> pd.DataFrame:
    with psycopg2.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=5432,
        database=os.environ["PROJECT_NAME"],
    ) as conn:
        query = "SELECT table_name"
        query += " FROM information_schema.tables"
        query += " WHERE table_schema='public'"
        query += " AND table_type='BASE TABLE'"
        query += " AND table_name LIKE '%{}%'".format(like)
        if unlike is not None:
            query += " AND table_name NOT ILIKE '%{}%'".format(unlike)
        query += " ORDER BY table_name; "
        df = pd.read_sql(query, con=conn)
        return df


def table_load(table_name: str, cols=None) -> pd.DataFrame:

    # with timer("table load ({})".format(table_name)):

    with psycopg2.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=5432,
        database=os.environ["PROJECT_NAME"],
    ) as conn:

        if cols is None:
            df = pd.read_sql(
                "SELECT * FROM {} ORDER BY index;".format(table_name), con=conn
            ).set_index("index")

        else:
            col_names_snake_case = [
                re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), col_name).lstrip(
                    "_"
                )
                for col_name in cols
            ]
            df = pd.read_sql(
                "SELECT index, {} FROM {} ORDER BY index;".format(
                    ", ".join(col_names_snake_case), table_name
                ),
                con=conn,
            ).set_index("index")
        return df


def table_write(table_name: str, df: pd.DataFrame):

    # Rename cols to snake_case
    df.columns = (
        pd.Series(df.columns)
        .map(
            lambda col: re.sub(
                "([A-Z])", lambda x: "_" + x.group(1).lower(), col
            ).lstrip("_")
        )
        .tolist()
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
