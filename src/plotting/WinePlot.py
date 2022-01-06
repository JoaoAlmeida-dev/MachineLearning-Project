import matplotlib.pyplot as plt
import numpy as np

from src.models.Wine import Wine
from typing import List

from src.models.WineSet import WineSet


def plotWineSet(wine_set: WineSet):
    row_count = int(Wine.N_VARIABLES ** 0.5) +1

    for header_index, header_string in enumerate(Wine.HEADERS):
        plt.subplot(row_count, row_count, header_index+1)
        plt.title(header_string)
        wine_set.wine_dataframe.loc[:,header_string].plot()

    plt.tight_layout()
    plt.show()
