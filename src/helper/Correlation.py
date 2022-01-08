import pandas

from src.constants.Constants import CORRELATIONS
from src.models.WineSet import WineSet


def correlate(wine_set: WineSet):
    dataframe = wine_set.wine_dataframe
    corr_dataframe = dataframe.corr(method="kendall")
    corr_dataframe.to_csv(path_or_buf=CORRELATIONS, sep=";")
    print(corr_dataframe)
