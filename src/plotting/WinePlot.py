import matplotlib.pyplot as plt
import numpy as np

from src.models.Wine import Wine
from typing import List

from src.models.WineSet import WineSet


def plotWineSet(wine_set: WineSet):
    row_count = int(Wine.N_VARIABLES ** 0.5) +1
    wine_set_transposed = wine_set.transposed

    for plot_index in range(1,Wine.N_VARIABLES+1):
        plt.subplot(row_count, row_count, plot_index)
        plt.plot(np.arange(0, len(wine_set.wine_list)), wine_set_transposed[plot_index - 1])
        plt.title(Wine.HEADERS[plot_index-1])
    plt.tight_layout()
    plt.show()
