import pandas
from pandas import DataFrame

from src.constants.Constants import HEADERS


def discretize(wine_set: DataFrame, num_bins: int):

    dataset_column_length: int = len(wine_set.columns)

    for collumn_index in range(dataset_column_length):
        collumn_name = HEADERS[collumn_index]

        column = wine_set.loc[:, collumn_name]
        column = pandas.cut(column, bins=num_bins, labels=range(num_bins))
