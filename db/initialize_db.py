import os

import pandas as pd

from db import exec_query, table_write

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
table_write(table_name="train", df=train)
table_write(table_name="test", df=test)
table_write(table_name="memo", df=memo)

# Create Index
exec_query("CREATE INDEX train_index on train (index);")
exec_query("CREATE INDEX test_index on test (index);")
exec_query("CREATE INDEX memo_index on memo (index);")

print("Done!!")
