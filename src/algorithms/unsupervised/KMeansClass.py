import numpy
from numpy import ndarray
from pandas import DataFrame
from sklearn import model_selection
from sklearn.cluster import KMeans

from src.constants.Constants import FEATURES, TARGET
import sklearn


class KMeansClass:
    algo_name: str = "KMeans"

    @classmethod
    def run(cls, wine_set: DataFrame) -> KMeans:
        features: ndarray = numpy.asarray(wine_set.drop(labels=TARGET, axis=1))
        labels: ndarray = numpy.asarray(wine_set[TARGET].values.tolist())

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: KMeans = KMeans()

        fit = algorithm.fit(features_train)
        predictions = algorithm.predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        # print(cls.algo_name, "predicitons", predictions)
        print(cls.algo_name, "accuracy", accuracy)
        return algorithm
