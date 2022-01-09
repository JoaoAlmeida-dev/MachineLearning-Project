import pandas
from pandas import DataFrame

from src.models.Wine import Wine
from src.models.WineSet import WineSet


def discretize(wine_set: WineSet, num_bins: int):
    dataframe = wine_set.wine_dataframe

    dataset_column_length: int = len(wine_set.wine_dataframe.columns)

    for collumn_index in range(dataset_column_length):
        collumn_name = Wine.HEADERS[collumn_index]

        dataframe.loc[:, collumn_name] = pandas.cut(dataframe.loc[:, collumn_name], bins=num_bins, labels=range(num_bins))

    wine_set.rebuild_from_dataframe()