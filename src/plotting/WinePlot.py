import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from src.constants.Constants import HEADERS, FEATURES, N_VARIABLES


def plot_hist_wine_set(wine_set: DataFrame, plt_figure_name: str, bins: int):
    row_count = int(N_VARIABLES ** 0.5) + 1
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)

    for header_index, header_string in enumerate(HEADERS):
        plt.subplot(row_count, row_count, header_index + 1)
        plt.title(header_string)
        wine_set.wine_dataframe.loc[:, header_string].plot.hist(bins=bins)

    plt.tight_layout()
    plt.show()


def plot_wine_set(wine_set: DataFrame, plt_figure_name: str):
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)
    wine_set.wine_dataframe.plot()
    plt.tight_layout()
    plt.show()


def plot_by_features(wine_set: DataFrame, plt_figure_name: str):
    row_count = int(len(FEATURES) ** 0.5) + 1
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)

    for header_index, header_string in enumerate(HEADERS):
        plt.subplot(row_count, row_count, header_index + 1)
        plt.title(header_string)
        wine_set.wine_dataframe.loc[:, header_string].plot.hist(bins=2)

    plt.tight_layout()
    plt.show()
