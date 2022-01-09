from matplotlib import pyplot as plt
from pandas import DataFrame

from src.constants.Constants import CORRELATIONS
from src.models.Wine import Wine
from src.models.WineSet import WineSet


def correlate(wine_set: WineSet) -> DataFrame:
    dataframe = wine_set.wine_dataframe
    corr_dataframe = dataframe.corr(method="kendall")
    #corr_dataframe.to_csv(path_or_buf=CORRELATIONS, sep=";")

    # Replaces the value of the cells that have the same attribute twice with 0.
    corr_dataframe.replace(to_replace=1, value=0, inplace=True)
    return corr_dataframe


def explore_correlation(correlations: DataFrame):
    correlations.replace(to_replace=1, value=0, inplace=True)
    print("Replaced")
    print(correlations.head())
    plt.imshow(correlations, cmap='bwr', interpolation='nearest')
    plt.title("correlations")
    plt.show()
    #compare two correlation
    if correlations.loc[Wine.TARGET, Wine.FEATURES[0]] < correlations.loc[Wine.TARGET, Wine.FEATURES[1]]:
        print(Wine.FEATURES[0])

