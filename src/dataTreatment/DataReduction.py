from pandas import DataFrame

from src.constants.Constants import FEATURES, TARGET


def reduce(wine_set: DataFrame, correlations: DataFrame):
    best_features_list: list = []

    for (row_index, row) in correlations.iterrows():
        if row_index != TARGET:
            idxmax = row.idxmax()
            row_max = row.max()
            if (idxmax, row_index, row_max) not in best_features_list and row_index != TARGET and idxmax != TARGET:
                best_features_list.append((row_index, idxmax, row_max))

    best_features_list.sort(key=lambda x: x[2], reverse=True)

    for i in range(2):
        feature_corr: (str, str, float) = best_features_list[i]
        corr_ = correlations.loc[TARGET, feature_corr[0]]
        feature_corr_ = correlations.loc[TARGET, feature_corr[1]]
        if corr_ < feature_corr_:
            wine_set.drop(labels=feature_corr[0], axis=1, inplace=True)
        else:
            wine_set.drop(labels=feature_corr[1], axis=1, inplace=True)
