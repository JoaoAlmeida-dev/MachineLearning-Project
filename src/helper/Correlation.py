from pandas import DataFrame


def correlate(wine_set: DataFrame) -> DataFrame:
    corr_dataframe = wine_set.corr(method="kendall")
    # Replaces the value of the cells that have the same attribute twice with 0.
    corr_dataframe.replace(to_replace=1, value=0, inplace=True)
    return corr_dataframe
