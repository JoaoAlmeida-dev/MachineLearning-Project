from matplotlib import pyplot as plt
from pandas import DataFrame

from src.constants.Constants import CORRELATIONS
from src.models.Wine import Wine
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
    plt.imshow(correlations, cmap='bwr', interpolation='nearest')
    plt.title("correlations")
    plt.show()
    #compare two correlation
    if correlations.loc[Wine.LABELS, Wine.FEATURES[0]] < correlations.loc[Wine.LABELS, Wine.FEATURES[1]]:
        print(Wine.FEATURES[0])

def merge(wine_set: WineSet, col_one=0, col_two=1):
    dataframe = wine_set.wine_dataframe
    new_col_name = Wine.FEATURES[col_one] + "_merge_with_" + Wine.FEATURES[col_two]
    dataframe[new_col_name] = (dataframe[Wine.FEATURES[col_one]] + dataframe[Wine.FEATURES[col_two]]) / 2

    print(dataframe.head)
