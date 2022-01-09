from pandas import DataFrame

from src.models.WineSet import WineSet


def correlate(wine_set: WineSet) -> DataFrame:
    dataframe = wine_set.wine_dataframe
    corr_dataframe = dataframe.corr(method="kendall")

    # Replaces the value of the cells that have the same attribute twice with 0.
    corr_dataframe.replace(to_replace=1, value=0, inplace=True)
    return corr_dataframe

