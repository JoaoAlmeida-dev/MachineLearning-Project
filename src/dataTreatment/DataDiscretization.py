import pandas
from pandas import DataFrame

from src.models.Wine import Wine
from src.models.WineSet import WineSet


def discretize(wine_set: WineSet, categories: DataFrame):
    dataframe = wine_set.wine_dataframe

    dataset_column_length: int = len(wine_set.wine_dataframe.columns)
    new_dataframe = DataFrame()
    for collumn_index in range(dataset_column_length):
        collumn_name = Wine.HEADERS[collumn_index]

        labels = categories.loc[:, collumn_name]
        name_ = dataframe.loc[:, collumn_name]
        new_dataframe[collumn_name] = pandas.cut(name_, bins=5, labels=range(5))
    return WineSet(new_dataframe)