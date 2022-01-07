
from sklearn import model_selection
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neural_network import MLPClassifier

from src.models.WineSet import WineSet
import  sklearn

class DBScanClass:
    algo_name:str = "DBscan"

    @classmethod
    def run(cls, wine_set: WineSet) -> DBSCAN:
        features = wine_set.features_as_ndarray
        labels = wine_set.labels_as_ndarray

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: DBSCAN = DBSCAN()

        predictions = algorithm.fit_predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        ##print(cls.multiLayerPercetronString, "predicitons", predictions)
        print(cls.algo_name, "accuracy", accuracy)
        print("ran", cls.algo_name)
        return algorithm

