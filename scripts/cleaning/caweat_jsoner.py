# -*- coding: UTF-8 -*-

import argparse
import pandas as pd

from collections import Counter, OrderedDict


COLUMNS = [
    "LANG",
    "VERSION",
    "BORN PLACE",
    # "FRUITS", # Not included because they are they are not part of IATs. Not published until used
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

WEATS = [
    # "FRUITS", # Not included because they are they are not part of IATs. Not published until used
    "WEAPONS", 
    "FLOWERS", 
    "INSTRUMENTS", 
    "INSECTS", 
    "PLEASANT", 
    "UNPLEASANT"
    ]

# Now a parameter.
# path_to_tsv = "data/tmp/CulturalAwareWEAT-es.tsv"

def _verify_column(label, records):
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


def _verify_data(df):
    # Verifying that all the records are correct
    for _, row in df.iterrows():
        print(_)
        for col in WEATS:
            _verify_column(col, row[col])
        print()

def tsv_to_json(input_file, output_file):
    df = pd.read_csv(input_file,
        sep="\t",
        header=0,
        usecols=COLUMNS,
        index_col="LANG")

    # Stripping string entries
    df_obj = df.select_dtypes('object')
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    
    _verify_data(df)

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
        orient="index",     #"records",
        # lines=True,
        indent=3
        )

def json_to_tsv(input_file, output_file):
    df = pd.read_json(input_file,
        orient="index",
        dtype=False)
   
   # Since LANG is an index in the json, it has to be explictly named before dumping
    df.index.name = 'LANG'
    print(df.head())
    # Dumping into new file
    df.to_csv(output_file, sep="\t")

def main(param):

    if param["mode"]:
        json_to_tsv(param["input"], param["output"])
    else:
        tsv_to_json(param["input"], param["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--input", dest="input", required=True, 
                        help = "Full/relative path to the input file")

    parser.add_argument('-o', "--output", dest="output", required=True,
                        help = "Full/relative path to the desired output file")

    parser.add_argument('-t', "--to-tsv", dest="mode", required=False, action='store_true',
                        help="Convert the input json file to tsv")

    parser.set_defaults(mode=False)

    arguments = parser.parse_args()

    param = OrderedDict()
    param["input"] = arguments.input
    param["output"] = arguments.output
    param["mode"] = arguments.mode

    main(param)

# From tsv to json
# $ python scripts/cleaning/caweat_jsoner.py -i data/tmp/CulturalAwareWEAT-es.tsv -o data/tmp/kk.json

# From json to tsv
# $ python scripts/cleaning/caweat_jsoner.py -t -i data/tmp/kk.json -o data/tmp/kk.tsv