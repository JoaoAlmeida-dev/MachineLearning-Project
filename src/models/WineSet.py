from typing import List

import numpy
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from src.dataTreatment.WineTranspose import wine_list_Transpose
from src.models.Wine import Wine


class WineSetIterator:
    def __init__(self, wineSet):
        self._wineSet = wineSet
        self._index = 0

    def __next__(self):
        if self._index < len(self._wineSet):
            result = self._wineSet[self._index]
            self._index += 1
            return result
        else:
            # End of Iteration
            raise StopIteration


class WineSet:

    def __init__(self, data):
        if isinstance(data, List):
            self.wine_list = data

            NDarray = []
            features_asNDArray = []
            labels_asNDArray = []
            for wine in data:
                NDarray.append(wine.asNDArray)
                features_asNDArray.append(wine.features_asNDArray)
                labels_asNDArray.append(wine.labels_asNDArray)

            self.wine_dataframe = pd.DataFrame(NDarray, columns=Wine.HEADERS)
            self.transposed = self.wine_dataframe.T

            self._asNDarray: ndarray = numpy.asarray(NDarray)
            self._features_asNDArray: ndarray = numpy.asarray(features_asNDArray)
            self._labels_asNDArray: ndarray = numpy.asarray(labels_asNDArray)

        elif isinstance(data, DataFrame):
            self.wine_list = [Wine.build_Series(series) for (index, series) in data.iterrows()]
            self.wine_dataframe = data
            self.transposed = data.T

            self._asNDarray: ndarray = numpy.asarray(data.values.tolist())
            self._features_asNDArray: ndarray = numpy.asarray(data[Wine.FEATURES].values.tolist())
            self._labels_asNDArray: ndarray = numpy.asarray(data[Wine.LABELS].values.tolist())


    def rebuild_from_dataframe(self):
        self.wine_list = [Wine.build_Series(series) for (index, series) in self.wine_dataframe.iterrows()]
        self.transposed = self.wine_dataframe.T

        self._asNDarray: ndarray = numpy.asarray(self.wine_dataframe.values.tolist())
        self._features_asNDArray: ndarray = numpy.asarray(self.wine_dataframe[Wine.FEATURES].values.tolist())
        self._labels_asNDArray: ndarray = numpy.asarray(self.wine_dataframe[Wine.LABELS].values.tolist())

    def __len__(self):
        return len(self.wine_list)

    def __iter__(self):
        return WineSetIterator(self)

    def __getitem__(self, item):
        return self.wine_list[item]

    def __str__(self):
        strRepr: str = "[\n"
        for wine in self.wine_list:
            strRepr += str(wine) + "\n"
        strRepr += "]\n"

        return strRepr

    def __repr__(self):
        return self.__str__()

    @property
    def features_asNDArray(self):
        return self._features_asNDArray

    @property
    def labels_asNDArray(self):
        return self._labels_asNDArray

    @property
    def asNDarray(self):
        return self._asNDarray
