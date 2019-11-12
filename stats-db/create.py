import pandas as pd
from tqdm import tqdm

from db import exec_query, find_table_name, table_load, table_write

drop_table_names = find_table_name(like="stats")["table_name"].to_list()
if len(drop_table_names) > 0:
    exec_query(
        "".join(
            [
                "DROP TABLE {};".format(drop_table_name)
                for drop_table_name in drop_table_names
            ]
        )
    )

table_names = []
table_names += find_table_name(like="train")["table_name"].to_list()
table_names += find_table_name(like="test")["table_name"].to_list()
table_names += find_table_name(like="cv_result")["table_name"].to_list()

for table_name in tqdm(table_names):
    df = table_load(table_name=table_name)
    stats = pd.concat(
        [
            df.dtypes.rename("dtype").astype(str).to_frame(),
            df.isnull().sum().rename("null_count").to_frame(),
            df.describe().T.rename(
                columns={"25%": "per_25", "50%": "per_50", "75%": "per_75"}
            ),
        ],
        axis=1,
        sort=False,
    )
    table_write(
        table_name="{}_stats".format(table_name), df=stats,
    )
