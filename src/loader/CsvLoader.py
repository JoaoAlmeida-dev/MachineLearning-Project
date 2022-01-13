import csv

import pandas


def load_raw_dataframe(filename: str, index_col=False):
    return pandas.read_csv(filename, sep=";", index_col=index_col)

