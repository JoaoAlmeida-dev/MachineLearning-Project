from pandas import DataFrame

from src.models.Wine import Wine
from src.models.WineSet import WineSet


def reduce(wine_set: WineSet, correlations: DataFrame) -> WineSet:
    best_features_list: list = []

    for feature in Wine.FEATURES:
        if (correlations.loc[feature].idxmax(), feature, correlations.loc[feature].max()) not in best_features_list:
            best_features_list.append((feature, correlations.loc[feature].idxmax(), correlations.loc[feature].max()))

    best_features_list.sort(key=lambda x: x[2], reverse=True)

    for i in range(2):
        feature_corr: (str, str, float) = best_features_list[i]
        corr_ = correlations.loc[Wine.TARGET, feature_corr[0]]
        feature_corr_ = correlations.loc[Wine.TARGET, feature_corr[1]]
        if corr_ < feature_corr_:
            wine_set.wine_dataframe.drop(labels=feature_corr[0], axis=1, inplace=True)
        else:
            wine_set.wine_dataframe.drop(labels=feature_corr[1], axis=1, inplace=True)

    wine_set.rebuild_from_dataframe()

