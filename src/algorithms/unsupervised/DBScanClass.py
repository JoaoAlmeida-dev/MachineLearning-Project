
from sklearn import model_selection
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neural_network import MLPClassifier

from src.models.WineSet import WineSet
import  sklearn

class DBScanClass:
    algoName:str = "DBscan"

    @classmethod
    def run(cls, wine_set: WineSet) -> DBSCAN:
        features = wine_set.features_asNDArray
        labels = wine_set.labels_asNDArray

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: DBSCAN = DBSCAN()

        predictions = algorithm.fit_predict(features_train)
        #accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        ##print(cls.multiLayerPercetronString, "predicitons", predictions)
        #print(cls.dBscanString, "accuracy", accuracy)
        print("ran", cls.algoName)
        return algorithm

