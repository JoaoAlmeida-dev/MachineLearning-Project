from typing import List

import matplotlib
from numpy import ndarray

from src.dataTreatment.WineTranspose import wine_list_Transpose
from src.models.WineSet import WineSet
from src.plotting.WinePlot import plotWineSet
from src.dataTreatment.RandomRemoval import RandomRemoval
from src.loader import CsvLoader
from src.models.Wine import Wine

RED_CSV = 'Resources/winequality-red.csv'
WHITE_CSV = 'Resources/winequality-white.csv'


def main():

    wine_list: List[Wine] = CsvLoader.load('%s' % RED_CSV, skip_header=True)
    print(wine_list)
    print("len=", len(wine_list))
    RandomRemoval(dataset=wine_list, removal_percentage=0.1)
    print("len=", len(wine_list))
    wine_list_transpose: ndarray = wine_list_Transpose(wine_list=wine_list)
    wine_set: WineSet = WineSet(wine_list)
    wine_set_transposed = wine_set.transposed


    plotWineSet(wine_set)


if __name__ == '__main__':
    main()
