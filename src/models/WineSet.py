from typing import List

import numpy
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from src.models.Wine import Wine


class WineSetIterator:
    def __init__(self, wine_set):
        self._wine_set = wine_set
        self._index = 0

    def __next__(self):
        if self._index < len(self._wine_set):
            result = self._wine_set[self._index]
            self._index += 1
            return result
        else:
            # End of Iteration
            raise StopIteration


class WineSet:
    wine_dataframe: DataFrame
    transposed: DataFrame
    _asNDarray: ndarray
    _features_asNDArray: ndarray
    _labels_asNDArray: ndarray

    def __init__(self, data):
        if isinstance(data, List):
            self.wine_list = data

            nd_array = []
            features_as_ndarray = []
            labels_as_ndarray = []
            for wine in data:
                nd_array.append(wine.as_ndarray)
                features_as_ndarray.append(wine.features_as_ndarray)
                labels_as_ndarray.append(wine.labels_as_ndarray)

            self.wine_dataframe = pd.DataFrame(nd_array, columns=Wine.HEADERS)
            self.transposed = self.wine_dataframe.T

            self._asNDarray: ndarray = numpy.asarray(nd_array)
            self._features_asNDArray: ndarray = numpy.asarray(features_as_ndarray)
            self._labels_asNDArray: ndarray = numpy.asarray(labels_as_ndarray)

        elif isinstance(data, DataFrame):
            self.wine_list = [Wine.build_from_series(series) for (index, series) in data.iterrows()]
            self.wine_dataframe = data
            self.transposed = data.T

            self._asNDarray: ndarray = numpy.asarray(data.values.tolist())
            self._features_asNDArray: ndarray = numpy.asarray(data[Wine.FEATURES].values.tolist())
            self._labels_asNDArray: ndarray = numpy.asarray(data[Wine.TARGET].values.tolist())


    def rebuild_from_dataframe(self):
        self.wine_list = [Wine.build_from_series(series) for (index, series) in self.wine_dataframe.iterrows()]
        self.transposed = self.wine_dataframe.T

        self._asNDarray: ndarray = numpy.asarray(self.wine_dataframe.values.tolist())
        self._features_asNDArray: ndarray = numpy.asarray(self.wine_dataframe[Wine.FEATURES].values.tolist())
        self._labels_asNDArray: ndarray = numpy.asarray(self.wine_dataframe[Wine.TARGET].values.tolist())

    def __len__(self):
        return len(self.wine_list)

    def __iter__(self):
        return WineSetIterator(self)

    def __getitem__(self, item):
        return self.wine_list[item]

    def __str__(self):
        str_repr: str = "[\n"
        for wine in self.wine_list:
            str_repr += str(wine) + "\n"
        str_repr += "]\n"

        return str_repr

    def __repr__(self):
        return self.__str__()

    @property
    def features_as_ndarray(self):
        return self._features_asNDArray

    @property
    def labels_as_ndarray(self):
        return self._labels_asNDArray

    @property
    def as_ndarray(self):
        return self._asNDarray
