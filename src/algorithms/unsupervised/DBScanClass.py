import numpy
from numpy import ndarray
from pandas import DataFrame
from sklearn import model_selection
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neural_network import MLPClassifier

from src.constants.Constants import FEATURES, TARGET
import  sklearn

class DBScanClass:
    algo_name:str = "DBscan"

    @classmethod
    def run(cls, wine_set: DataFrame) -> DBSCAN:
        features: ndarray = numpy.asarray(wine_set[:, wine_set.columns != TARGET].values.tolist())
        labels: ndarray = numpy.asarray(wine_set[TARGET].values.tolist())

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: DBSCAN = DBSCAN()

        predictions = algorithm.fit_predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        ##print(cls.multiLayerPercetronString, "predicitons", predictions)
        print(cls.algo_name, "accuracy", accuracy)
        print("ran", cls.algo_name)
        return algorithm

