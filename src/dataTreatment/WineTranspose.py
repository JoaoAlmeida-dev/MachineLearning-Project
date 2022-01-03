from typing import List

import numpy
from numpy import ndarray

from src.models.Wine import Wine


def wine_list_Transpose(wine_list) -> ndarray:
    # transposed_list:List[List[float]] =[ [] for range(0,Wine.number_of_variabes)]
    wine_dataset_ndarray_list: List[ndarray] = []
    transposed_list: ndarray
    for wine_bottle in wine_list:
        wine_dataset_ndarray_list.append(wine_bottle.asNDArray)

    return numpy.asarray(wine_dataset_ndarray_list).T
