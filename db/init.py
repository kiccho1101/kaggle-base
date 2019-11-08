import os

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Drop table train, test
with psycopg2.connect(
    user=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host=os.environ["POSTGRES_HOST"],
    port=5432,
    database=os.environ["PROJECT_NAME"],
) as conn, conn.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS train;")
    cur.execute("DROP TABLE IF EXISTS test;")
    cur.execute("DROP TABLE IF EXISTS feature;")
    conn.commit()

# Read data
train = pd.read_csv(os.environ["PROJECT_DIR"] + "/input/train.csv")
test = pd.read_csv(os.environ["PROJECT_DIR"] + "/input/test.csv")
feature = pd.read_csv(os.environ["PROJECT_DIR"] + "/input/feature.csv")

# Insert train, test data into DB
engine = create_engine(
    "postgresql://{}:{}@{}:5432/{}".format(
        os.environ["POSTGRES_USER"],
        os.environ["POSTGRES_PASSWORD"],
        os.environ["POSTGRES_HOST"],
        os.environ["PROJECT_NAME"],
    )
)
train.to_sql("train", engine)
test.to_sql("test", engine)
feature.to_sql("feature", engine)
