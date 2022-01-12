import csv

import pandas

from typing import List, Union, Any

from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
from pandas.io.parsers import TextFileReader


def load_raw_dataframe(filename: str, index_col=False):
    return pandas.read_csv(filename, sep=";", index_col=index_col)

