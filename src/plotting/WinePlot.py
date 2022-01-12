import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from pandas import DataFrame

from src.constants.Constants import HEADERS, FEATURES, N_VARIABLES, TARGET


def plot_hist_wine_set(wine_set: DataFrame, plt_figure_name: str, bins: int):
    row_count = int(N_VARIABLES ** 0.5) + 1
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)

    for header_index, header_string in enumerate(HEADERS):
        plt.subplot(row_count, row_count, header_index + 1)
        plt.title(header_string)
        wine_set.loc[:, header_string].plot.hist(bins=bins)

    plt.tight_layout()
    plt.show()


def plot_wine_set(wine_set: DataFrame, plt_figure_name: str):
    wine_set.plot()
    plt.title(plt_figure_name)
    plt.tight_layout()
    plt.show()


def plot_by_features(wine_set: DataFrame, plt_figure_name: str):
    row_count = int(len(FEATURES) ** 0.5) + 1
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)

    for header_index, header_string in enumerate(HEADERS):
        plt.subplot(row_count, row_count, header_index + 1)
        plt.title(header_string)
        wine_set.loc[:, header_string].plot.hist(bins=2)

    plt.tight_layout()
    plt.show()


def plot_decision_regions_algo(wine_set: DataFrame, algorithm):
    features_df: DataFrame = wine_set.drop(labels=TARGET, axis=1)
    labels_df: DataFrame = wine_set[TARGET]

    plot_decision_regions(features_df.values, labels_df.values, clf=algorithm, legend=2, feature_index=[0, 1],
                          filler_feature_values={2: wine_set.iloc[:, 2].mean(), 3: wine_set.iloc[:, 3].mean(),
                                                 4: wine_set.iloc[:, 4].mean(), 5: wine_set.iloc[:, 5].mean(),
                                                 6: wine_set.iloc[:, 6].mean(), 7: wine_set.iloc[:, 7].mean(),
                                                 8: wine_set.iloc[:, 8].mean(), 9: wine_set.iloc[:, 9].mean(),
                                                 10: wine_set.iloc[:, 10].mean()
                                                 },
                          filler_feature_ranges={2: wine_set.iloc[:, 2].max(), 3: wine_set.iloc[:, 3].max(),
                                                 4: wine_set.iloc[:, 4].max(), 5: wine_set.iloc[:, 5].max(),
                                                 6: wine_set.iloc[:, 6].max(), 7: wine_set.iloc[:, 7].max(),
                                                 8: wine_set.iloc[:, 8].max(), 9: wine_set.iloc[:, 9].max(),
                                                 10: wine_set.iloc[:, 10].max()})

    plt.show()
