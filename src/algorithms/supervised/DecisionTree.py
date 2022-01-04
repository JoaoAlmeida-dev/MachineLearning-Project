from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier

from src.models.WineSet import WineSet
import  sklearn

class DecisionTree:
    algoName:str = "DecisionTree"

    @classmethod
    def run(cls, wine_set: WineSet) -> DecisionTreeClassifier:
        features = wine_set.features_asNDArray
        labels = wine_set.labels_asNDArray

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: DecisionTreeClassifier = DecisionTreeClassifier().fit(features_train, labels_train)

        predictions = algorithm.predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        #print(cls.decisionTreeString,"predicitons", predictions)
        print(cls.algoName, "accuracy", accuracy)
        print("ran", cls.algoName)
        return algorithm
