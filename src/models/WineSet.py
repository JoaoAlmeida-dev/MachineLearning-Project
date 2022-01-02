from typing import List

import numpy
from numpy import ndarray

from src.dataTreatment.WineTranspose import wine_list_Transpose
from src.models.Wine import Wine


class WineSet:

    def __init__(self, wine_list: List[Wine]):
        self.wine_list = wine_list
        self.transposed = wine_list_Transpose(wine_list)

