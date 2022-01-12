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
    def run(cls, wine_set: DataFrame) -> DecisionTreeClassifier:
        features: ndarray = numpy.asarray(wine_set.drop(labels=TARGET, axis=1))
        labels: ndarray = numpy.asarray(wine_set[TARGET].values.tolist())

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        labels_train = labels_train.ravel()
        algorithm: DecisionTreeClassifier = DecisionTreeClassifier()
        algorithm.fit(features_train, labels_train)

        predictions = algorithm.predict(features_test)
        accuracy = sklearn.metrics.accuracy_score(labels_test, predictions)
        # print(cls.decisionTreeString,"predicitons", predictions)
        print(cls.algo_name, "accuracy", accuracy)
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(algorithm,
                           feature_names=FEATURES,
                           class_names=HEADERS,
                           filled=True)
        fig.savefig("decision_tree.png")


        return algorithm
