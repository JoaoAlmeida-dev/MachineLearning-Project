from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

from src.models.WineSet import WineSet
import  sklearn

class MultiLayerPercetron:
    multiLayerPercetronString:str = "MultiLayerPercetron"

    @classmethod
    def run(cls, wine_set: WineSet):
        features = wine_set.features_asNDArray
        labels = wine_set.labels_asNDArray

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        knn = MLPClassifier().fit(features_train, labels_train)

        predictions = knn.predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        #print(cls.multiLayerPercetronString, "predicitons", predictions)
        print(cls.multiLayerPercetronString, "accuracy", accuracy)