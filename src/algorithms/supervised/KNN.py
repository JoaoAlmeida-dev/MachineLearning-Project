import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn import model_selection
from src.main import RED_CSV


def main():
    data = pd.read_csv("../../../Resources/winequality-red.csv", sep=";")

    features = data[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                     'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
                     ]]
    labels = data[['quality']]
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
