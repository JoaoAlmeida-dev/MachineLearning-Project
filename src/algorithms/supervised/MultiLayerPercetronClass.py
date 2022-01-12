import numpy
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from numpy import ndarray
from pandas import DataFrame
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

from src.constants.Constants import FEATURES, TARGET
import sklearn

class MultiLayerPercetronClass:
    algo_name:str = "MultiLayerPercetron"

    @classmethod
    def run(cls, wine_set: DataFrame) -> MLPClassifier:
        features: ndarray = numpy.asarray(wine_set.drop(labels=TARGET, axis=1))
        labels: ndarray = numpy.asarray(wine_set[TARGET].values.tolist())

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        labels_train = labels_train.ravel()
        algorithm: MLPClassifier = MLPClassifier(max_iter=3000)
        algorithm.fit(features_train, labels_train)

        predictions = algorithm.predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        #print(cls.multiLayerPercetronString, "predicitons", predictions)
        print(cls.algo_name, "accuracy", accuracy)
        return algorithm
