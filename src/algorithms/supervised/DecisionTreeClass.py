import numpy
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn import model_selection, tree
from sklearn.tree import DecisionTreeClassifier
import sklearn

from src.constants.Constants import FEATURES, TARGET, HEADERS


class DecisionTreeClass:
    algo_name: str = "DecisionTree"

    @classmethod
    def run(cls, features_train, features_test, labels_train, labels_test) -> DecisionTreeClassifier:

        algorithm: DecisionTreeClassifier = DecisionTreeClassifier()
        algorithm.fit(features_train, labels_train)

        predictions = algorithm.predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        # print(cls.decisionTreeString,"predicitons", predictions)
        print(cls.algo_name, "accuracy", accuracy)


        return algorithm
