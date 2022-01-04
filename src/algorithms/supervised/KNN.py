import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

from src.models.Wine import Wine
from src.models.WineSet import WineSet


class KNN:
    algoName:str = "KNN"

    @classmethod
    def run(cls, wine_set: WineSet) -> KNeighborsClassifier:
        features = wine_set.features_asNDArray
        labels = wine_set.labels_asNDArray

        features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels,
                                                                                                    test_size=0.2)
        algorithm: KNeighborsClassifier = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform").fit(features_train, labels_train)

        predictions = algorithm.predict(features_test)
        accuracy = metrics.accuracy_score(labels_test, predictions)
        #print(cls.knnString,"predicitons", predictions)
        print(cls.algoName, "accuracy", accuracy)
        print("ran", cls.algoName)
        return algorithm




def main():
    data = pd.read_csv("../../../Resources/winequality-red.csv", sep=";")

    features = data[Wine.FEATURES]
    labels = data[Wine.LABELS]

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
