import math
from typing import Callable

import pandas
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from src.models.Wine import Wine
from src.models.WineSet import WineSet


def normalize_set_log(wine_set: WineSet):
    def _log_item(item: float, column_name: str) -> float:
        if item <= 0:
            item = 0.01
        log = math.log(item)
        return log

    _normalize(wine_set=wine_set, normalizing_operation=_log_item)


def normalize_set_range(wine_set: WineSet, range_dataframe: DataFrame, set_name: str):
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


def normalize_set_mean(wine_set: WineSet):
    dataframe = wine_set.wine_dataframe
    variables = Wine.FEATURES
    target = Wine.TARGET
    x = dataframe.loc[:, variables].values
    y = dataframe.loc[:, target].values
    x = StandardScaler().fit_transform(x)
    x = pandas.DataFrame(x)
    x.columns = variables
    x[target] = y
    wine_set.wine_dataframe = x
    wine_set.rebuild_from_dataframe()


def _normalize(wine_set: WineSet, normalizing_operation: Callable):
    dataframe: DataFrame = wine_set.wine_dataframe

    dataset_row_length: int = len(wine_set.wine_dataframe)
    dataset_column_length: int = len(wine_set.wine_dataframe.columns) - 1
    for row_index in range(dataset_row_length):
        for column_index in range(dataset_column_length):
            column_name = Wine.FEATURES[column_index]
            original_value = dataframe.loc[row_index, column_name]

            dataframe.loc[row_index, column_name] = normalizing_operation(item=original_value,
                                                                          column_name=column_name)
    wine_set.rebuild_from_dataframe()
