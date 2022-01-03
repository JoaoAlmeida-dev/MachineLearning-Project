from typing import List

import numpy
from numpy import ndarray
from pandas import DataFrame

from src.dataTreatment.WineTranspose import wine_list_Transpose
from src.models.Wine import Wine


class WineSet:

    def __init__(self, wine_list: List[Wine] = None, wine_dataframe: DataFrame = None):
        if wine_list is not None:
            self.wine_list = wine_list
            self._len = len(wine_list)
            self.transposed = wine_list_Transpose(wine_list)
            NDarray = []
            features_asNDArray = []
            labels_asNDArray = []
            for wine in wine_list:
                NDarray.append(wine.asNDArray)
                features_asNDArray.append(wine.features_asNDArray)
                labels_asNDArray.append(wine.labels_asNDArray)
            self._asNDarray:ndarray = numpy.asarray(NDarray)
            self._features_asNDArray:ndarray = numpy.asarray(features_asNDArray)
            self._labels_asNDArray:ndarray = numpy.asarray(labels_asNDArray)

        elif wine_dataframe is not None:
            # wine_list =
            pass

    def __len__(self):
        return self._len

    @property
    def features_asNDArray(self):
        return self._features_asNDArray

    @property
    def labels_asNDArray(self):
        return self._labels_asNDArray

    @property
    def asNDarray(self):
        return self._asNDarray
