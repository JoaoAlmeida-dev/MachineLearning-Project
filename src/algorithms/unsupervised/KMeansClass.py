from sklearn import model_selection
from sklearn.cluster import KMeans

from src.models.WineSet import WineSet
import  sklearn

class KMeansClass:
    algoName:str = "KMeans"

    @classmethod
    def run(cls, wine_set: WineSet) -> KMeans:
        features = wine_set.features_asNDArray
        labels = wine_set.labels_asNDArray

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: KMeans= KMeans()

        fit = algorithm.fit(features_train)
        predictions = algorithm.predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        print(cls.algoName, "predicitons", predictions)
        print(cls.algoName, "accuracy", accuracy)
        print("ran", cls.algoName)
        return algorithm
