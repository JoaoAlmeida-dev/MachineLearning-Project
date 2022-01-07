import matplotlib.pyplot as plt
import numpy as np

from src.models.Wine import Wine
from typing import List

from src.models.WineSet import WineSet


def plot_wine_set(wine_set: WineSet, plt_figure_name: str):
    row_count = int(Wine.N_VARIABLES ** 0.5) + 1
    fig = plt.gcf()
    fig.canvas.set_window_title(plt_figure_name)

    for header_index, header_string in enumerate(Wine.HEADERS):
        plt.subplot(row_count, row_count, header_index + 1)
        plt.title(header_string)
        plt.hist(wine_set.wine_dataframe.loc[:, header_string], bins=20)

    plt.tight_layout()
    plt.show()
