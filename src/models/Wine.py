from operator import __getitem__
from typing import List, Set

import numpy
import numpy as np
from numpy import ndarray
from pandas import Series


class Wine:
    # region CONSTANTS
    HEADERS: List[str] = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                          "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                          "quality", ]
    FEATURES: List[str] = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                           "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                           ]
    LABELS: List[str] = ["quality"]

    N_VARIABLES: int = len(HEADERS) - 1

    # endregion

    # region VARIABLES

    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    _quality: float
    _ndarray: ndarray

    _features_ndarray: ndarray
    _labels_ndarray: ndarray

    # endregion
    @classmethod
    def build_Series(cls, series: Series):

        return Wine(fixed_acidity=series[0],
                    volatile_acidity=series[1],
                    citric_acid=series[2],
                    residual_sugar=series[3],
                    chlorides=series[4],
                    free_sulfur_dioxide=series[5],
                    total_sulfur_dioxide=series[6],
                    density=series[7],
                    pH=series[8],
                    sulphates=series[9],
                    alcohol=series[10],
                    output=series[11],
                    )

    def __init__(self, fixed_acidity: float = -1, volatile_acidity: float = -1, citric_acid: float = -1,
                 residual_sugar: float = -1,
                 chlorides: float = -1, free_sulfur_dioxide: float = -1, total_sulfur_dioxide: float = -1,
                 density: float = -1, pH: float = -1,
                 sulphates: float = -1, alcohol: float = -1, output: float = 0):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        self._quality = output

        self._ndarray = numpy.asarray(
            [self.fixed_acidity, self.volatile_acidity, self.citric_acid, self.residual_sugar, self.chlorides,
             self.free_sulfur_dioxide, self.total_sulfur_dioxide, self.density, self.pH, self.sulphates, self.alcohol,
             self._quality])
        self._features_ndarray = numpy.asarray(
            [self.fixed_acidity, self.volatile_acidity, self.citric_acid, self.residual_sugar, self.chlorides,
             self.free_sulfur_dioxide, self.total_sulfur_dioxide, self.density, self.pH, self.sulphates, self.alcohol])
        self._labels_ndarray = numpy.asarray([self._quality, ])

    @property
    def labels_asNDArray(self):
        return self._labels_ndarray

    @property
    def features_asNDArray(self):
        return self._features_ndarray

    @property
    def asNDArray(self):
        return self._ndarray

    @property
    def quality(self):
        return self._quality

    def __getitem__(self, index):
        if index == 0:
            return self.fixed_acidity
        elif index == 1:
            return self.volatile_acidity
        elif index == 2:
            return self.citric_acid
        elif index == 3:
            return self.residual_sugar
        elif index == 4:
            return self.chlorides
        elif index == 5:
            return self.free_sulfur_dioxide
        elif index == 6:
            return self.total_sulfur_dioxide
        elif index == 7:
            return self.density
        elif index == 8:
            return self.pH
        elif index == 9:
            return self.sulphates
        elif index == 10:
            return self.alcohol

    def __str__(self):
        return "[{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11}]".format(str(self.fixed_acidity),
                                                                            str(self.volatile_acidity),
                                                                            str(self.citric_acid),
                                                                            str(self.residual_sugar),
                                                                            str(self.chlorides),
                                                                            str(self.free_sulfur_dioxide),
                                                                            str(self.total_sulfur_dioxide),
                                                                            str(self.density),
                                                                            str(self.pH), str(self.sulphates),
                                                                            str(self.alcohol),
                                                                            str(self.quality))

    def __repr__(self):
        return self.__str__()
