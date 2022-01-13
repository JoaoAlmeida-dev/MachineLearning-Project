import numpy
from numpy import ndarray
from pandas import DataFrame
from sklearn import model_selection
from sklearn.cluster import AgglomerativeClustering
from sklearn.neural_network import MLPClassifier

from src.constants.Constants import FEATURES, TARGET
import  sklearn

class AgglomerativeHierarchicalClusteringClass:
    algo_name:str = "AgglomerativeHierarchicalClustering"

    @classmethod
    def run(cls, wine_set: DataFrame) -> AgglomerativeClustering:
        features: ndarray = numpy.asarray(wine_set.drop(labels=TARGET, axis=1))
        labels: ndarray = numpy.asarray(wine_set[TARGET].values.tolist())

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: AgglomerativeClustering = AgglomerativeClustering(n_clusters=5)

        predictions = algorithm.fit_predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        ##print(cls.multiLayerPercetronString, "predicitons", predictions)
        print(cls.algo_name, "accuracy", accuracy)
        return algorithm ,accuracy
