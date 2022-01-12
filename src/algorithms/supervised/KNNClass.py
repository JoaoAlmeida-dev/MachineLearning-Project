import numpy
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn import neighbors, metrics
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from src.constants.Constants import FEATURES, TARGET
from mlxtend.plotting import plot_decision_regions

class KNNClass:
    algo_name: str = "KNN"

    @classmethod
    def run(cls, wine_set: DataFrame) -> KNeighborsClassifier:
        features: ndarray = numpy.asarray(wine_set.drop(labels=TARGET, axis=1))
        labels: ndarray = numpy.asarray(wine_set[TARGET].values.tolist())

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: KNeighborsClassifier = neighbors.KNeighborsClassifier(n_neighbors=1)
        algorithm.fit(features_train, labels_train.ravel())

        predictions = algorithm.predict(features_test)
        accuracy = metrics.accuracy_score(labels_test, predictions)
        print(cls.algo_name, "accuracy", accuracy)
        return algorithm




def main():
    data = pd.read_csv("../../../res/winequality-red.csv", sep=";")

    features = data[FEATURES]
    labels = data[TARGET]

    features = np.array(features)
    labels = np.array(labels)

    print("features", features)
    print("labels", labels)

    # create model
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)

    features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                test_size=0.2)
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform").fit(features_train, labels_train)

    predictions = knn.predict(features_test)
    accuracy = metrics.accuracy_score(labels_test, predictions)
    print("predicitons", predictions)
    print("accuracy", accuracy)


if __name__ == '__main__':
    main()
