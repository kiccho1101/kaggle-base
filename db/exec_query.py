import os

import psycopg2


def exec_query(query=""):

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
