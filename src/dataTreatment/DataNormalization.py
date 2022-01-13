import math
from typing import Callable

import pandas
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from src.constants.Constants import TARGET, FEATURES


def normalize_set_log(wine_set: DataFrame):
    def _log_item(item: float, column_name: str) -> float:
        if item <= 0:
            item = 0.01
        log = math.log(item)
        return log

    _normalize(wine_set=wine_set, normalizing_operation=_log_item)


def normalize_set_range(wine_set: DataFrame, range_dataframe: DataFrame, set_name: str):
    def _range_item(item: float, column_name: str) -> float:
        min_column: str = column_name + "_min"
        max_column: str = column_name + "_max"

        min_value: float = range_dataframe.loc[set_name, min_column]
        max_value: float = range_dataframe.loc[set_name, max_column]
        if item > max_value:
            item = max_value
        elif item < min_value:
            item = min_value

        new_value: float = (item - min_value) / (max_value - min_value)

        return new_value

    _normalize(wine_set=wine_set, normalizing_operation=_range_item)


def normalize_standard_scaler(wine_set: DataFrame):

    variables = FEATURES
    target = TARGET
    x = wine_set.loc[:, variables].values
    y = wine_set.loc[:, target].values
    x = StandardScaler().fit_transform(x)
    x = pandas.DataFrame(x)
    x.columns = variables
    x[target] = y
    wine_set = x


def _normalize(wine_set: DataFrame, normalizing_operation: Callable):
    dataset_row_length: int = len(wine_set)
    dataset_column_length: int = len(wine_set.columns) - 1
    for row_index in range(dataset_row_length):
        for column_index in range(dataset_column_length):
            column_name = FEATURES[column_index]
            original_value = wine_set.loc[row_index, column_name]

            wine_set.loc[row_index, column_name] = normalizing_operation(item=original_value,
                                                                         column_name=column_name)
