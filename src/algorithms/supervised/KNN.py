import pandas as pd
from sklearn import neighbors, metrics

from src.main import RED_CSV


def main():
    data = pd.read_csv("/home/joao/PycharmProjects/pythonProject/Resources/winequality-red.csv",sep=";")
    print(data.head())

    features = [["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                 "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                 ]]
    labels = [["output"]]

    knn = neighbors.KNeughboursClassifier(n_neighbors=5,weights="uniform")
    


if __name__ == '__main__':
    main()


