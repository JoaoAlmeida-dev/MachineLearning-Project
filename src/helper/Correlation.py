import pandas
from pandas import DataFrame

from src.constants.Constants import CORRELATIONS
from src.models.WineSet import WineSet


def correlate(wine_set: WineSet):
    dataframe = wine_set.wine_dataframe
    corr_dataframe = dataframe.corr(method="kendall")
    corr_dataframe.to_csv(path_or_buf=CORRELATIONS, sep=";")
    print(corr_dataframe)


def explore_correlation(correlations: DataFrame):
    correlations.replace(to_replace=1, value=0, inplace=True)
    print("Replaced")
    print(correlations.head())
