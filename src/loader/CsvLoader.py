import csv

import pandas

from typing import List, Union, Any

from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
from pandas.io.parsers import TextFileReader

from src.models.Wine import Wine
from src.models.WineSet import WineSet


def load_raw_dataframe(filename: str, index_col=False):
    return pandas.read_csv(filename, sep=";", index_col=index_col)


def load_dataframe(filename: str) -> WineSet:
    file_df: Union[Union[TextFileReader, Series, DataFrame, None, NDFrame], Any] = load_raw_dataframe(filename=filename)
    # file_df[Wine.HEADERS] = file_df[Wine.HEADERS].astype(float)
    return WineSet(file_df)


def load_list(filename: str, skip_header: bool = True) -> WineSet:
    wine_list: List[Wine] = []

    # opening the CSV file
    with open(filename, mode='r') as file:
        # reading the CSV file
        csv_file = csv.reader(file)
        if skip_header: next(csv_file)
        # displaying the contents of the CSV file
        for lines in csv_file:
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol, output = \
                lines[0].split(";")
            wine = Wine(fixed_acidity=float(fixed_acidity),
                        volatile_acidity=float(volatile_acidity),
                        citric_acid=float(citric_acid),
                        residual_sugar=float(residual_sugar),
                        chlorides=float(chlorides),
                        free_sulfur_dioxide=float(free_sulfur_dioxide),
                        total_sulfur_dioxide=float(total_sulfur_dioxide),
                        density=float(density),
                        ph=float(ph),
                        sulphates=float(sulphates),
                        alcohol=float(alcohol), output=float(output))
            wine_list.append(wine)

    return WineSet(data=wine_list)
