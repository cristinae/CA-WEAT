# -*- coding: UTF-8 -*-

import pandas as pd

from collections import Counter
# from caweat_inspector import Inspector


COLUMNS = [
    "LANG",
    "BORN PLACE",
    "FRUITS", # Not in v1 of the dataset???
    "WEAPONS",
    "FLOWERS", 
    "INSTRUMENTS", 
    "INSECTS", 
    "PLEASANT", 
    "UNPLEASANT"
    ]   

EXTRA_COLUMNS = [
    "TYPE",
    "WHO",	
]

path_to_tsv = "data/tmp/CulturalAwareWEAT-es.tsv"

def verify_column(label, records):
    lst = [kk.strip() for kk in records.split(",")]
    items = set(lst)
    if len(items) != 25:
        print("UNIQUE ITEMS", len(items))
        print(label)
        if len(lst) > len(items):
            print("duplicated")
            # # not efficient, but useful to spot duplicates
            # tmp = set()
            # for k in lst:
            print([k for k,v in Counter(lst).items() if v>1])
                
        # print(label, "records:", len(items))
        # print("ALL", lst)
        # print("DED", items)
    
    


df = pd.read_csv(path_to_tsv,
        sep="\t",
        header=0,
        usecols=COLUMNS,
        index_col="LANG")

# Stripping string entries
df_obj = df.select_dtypes('object')
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
# print (df)


# Verifying that all the records are correct
# for index, row in df.iterrows():
#     print(index)  #, row)
#     for col in row:
#         print(col)
# for series_name, series in df.items():
#     # print(series_name)
#     print("kk")
#     print(series)
# for name, values in df.iteritems():
#     print('{name}: {value}'.format(name=name, value=values[0]))
# for (idx, row) in df.iterrows():
#     print(df[idx])
for _, row in df.iterrows():
    print(_)
    for col in ["FRUITS", "WEAPONS", "FLOWERS", "INSTRUMENTS", "INSECTS", "PLEASANT", "UNPLEASANT"]:
        verify_column(col, row[col])
    print()

# Adding the two additional columns
# I use insert instead of e.g., 
# df["TYPE"] = ["original"] * len(df)
# df["WHO"] = ["anonymous"] * len(df)
# to force the additional columns to appear first (as in the tsv in v1.0)
df.insert(loc=0, column="WHO", value=["anonymous"] * len(df))
df.insert(loc=0, column="TYPE", value=["original"] * len(df))


# Dumping into new file
df.to_json("data/tmp/kk.json",
    force_ascii=False,
    orient="index",             #"records",
    # lines=True,
    indent=3
    )