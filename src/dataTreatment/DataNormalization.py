import math
from typing import Callable

from pandas import DataFrame

from src.models.Wine import Wine
from src.models.WineSet import WineSet


def normalize_set_log(wine_set: WineSet):
    def _log_item(item: float, collumn_name: str) -> float:
        if item <= 0:
            item = 0.01
        log = math.log(item)
        return log

    _normalize(wine_set=wine_set, normalizing_operation=_log_item)


def normalize_set_range(wine_set: WineSet, range: DataFrame):
    def _range_item(item: float, collumn_name: str) -> float:
        min_column: str = collumn_name + "_min"
        max_column: str = collumn_name + "_max"

        min_value: float = range.loc[0,min_column]
        max_value: float = range.loc[0,max_column]
        if item > max_value:
            item = max_value
        elif item < min_value:
            item = min_value

        new_value: float = (item - min_value) / (max_value - min_value)

        return new_value

    _normalize(wine_set=wine_set, normalizing_operation=_range_item)


def _normalize(wine_set: WineSet, normalizing_operation: Callable):
    dataframe: DataFrame = wine_set.wine_dataframe

    dataset_row_length: int = len(wine_set.wine_dataframe)
    dataset_column_length: int = len(wine_set.wine_dataframe.columns) - 1
    for row_index in range(dataset_row_length):
        for collumn_index in range(dataset_column_length):
            collumn_name = Wine.FEATURES[collumn_index]
            original_value = dataframe.loc[row_index, collumn_name]

            dataframe.loc[row_index, collumn_name] = normalizing_operation(item=original_value,
                                                                           collumn_name=collumn_name)
    wine_set.rebuild_from_dataframe()
