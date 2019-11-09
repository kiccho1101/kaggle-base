import os

import pandas as pd

from db.exec_query import exec_query
from db.table_write import table_write

print("Initializing Database...")

# Drop tables if they exist
exec_query("DROP TABLE IF EXISTS train;")
exec_query("DROP TABLE IF EXISTS test;")
exec_query("DROP TABLE IF EXISTS memo;")

# Read data
train = pd.read_csv(os.environ["PROJECT_DIR"] + "/input/train.csv")
test = pd.read_csv(os.environ["PROJECT_DIR"] + "/input/test.csv")
memo = pd.read_csv(os.environ["PROJECT_DIR"] + "/input/memo.csv")

# Insert train, test data into DB
table_write(df=train, table_name="train")
table_write(df=test, table_name="test")
table_write(df=memo, table_name="memo")

print("Done!!")
