from operator import __getitem__
from typing import List, Set

import numpy
import numpy as np
from numpy import ndarray


class Wine:
    N_VARIABLES: int = 11
    HEADERS: List[str] = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                         "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                         "output",]
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
    output: float

    _ndarray: ndarray

    def __init__(self, fixed_acidity: float, volatile_acidity: float, citric_acid: float, residual_sugar: float,
                 chlorides: float, free_sulfur_dioxide: float, total_sulfur_dioxide: float, density: float, pH: float,
                 sulphates: float, alcohol: float, output: float = 0):
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
        self.output = output
        self._ndarray = numpy.asarray([
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.residual_sugar,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.pH,
            self.sulphates,
            self.alcohol
        ])

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

    def toNdArray(self):
        return self._ndarray

    def __str__(self):
        return "{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}".format(str(self.fixed_acidity), str(self.volatile_acidity),
                                                               str(self.citric_acid), str(self.residual_sugar),
                                                               str(self.chlorides), str(self.free_sulfur_dioxide),
                                                               str(self.total_sulfur_dioxide), str(self.density),
                                                               str(self.pH), str(self.sulphates), str(self.alcohol),
                                                               str(self.output))

    def __repr__(self):
        return self.__str__()
